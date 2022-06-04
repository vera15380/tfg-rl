"""
Policy module
"""

from typing import List

from atcenv import calcs
from atcenv.definitions import *


def prediction_conflicts(max_dt, i, j, actions, env):
    """
    Predicts the future position until dt seconds, maintaining the current speed and track
    :param max_dt: prediction look-ahead time (in seconds)
    :param i: flight i identifier
    :param j: flight j identifier
    :param actions: vector of flight actions
    :param env: Environment
    X and Y Speed components (in kt)
    """
    global track_i, track_j

    if actions[i] == 0:
        track_i = env.flights[i].track
    elif actions[i] == 1:
        track_i = env.flights[i].bearing + u.circle / 8
        track_i = track_i % u.circle
    elif actions[i] == 2:
        track_i = env.flights[i].bearing - u.circle / 8
        track_i = track_i % u.circle
    elif actions[i] == 3:
        track_i = env.flights[i].bearing + u.circle / 4
        track_i = track_i % u.circle

    if actions[j] == 0:
        track_j = env.flights[j].track
    elif actions[j] == 1:
        track_j = env.flights[j].bearing + u.circle / 8
        track_j = track_j % u.circle
    elif actions[j] == 2:
        track_j = env.flights[j].bearing - u.circle / 8
        track_j = track_j % u.circle
    elif actions[j] == 3:
        track_j = env.flights[j].bearing + u.circle / 4
        track_j = track_j % u.circle

    WillBeConflict = False

    dx_i = env.flights[i].airspeed * math.sin(track_i)
    dy_i = env.flights[i].airspeed * math.cos(track_i)

    dx_j = env.flights[j].airspeed * math.sin(track_j)
    dy_j = env.flights[j].airspeed * math.cos(track_j)

    dt_step = env.dt
    for dt_step in range(max_dt):
        position_i = Point(
            [env.flights[i].position.x + dx_i * dt_step, env.flights[i].position.y + dy_i * dt_step])
        position_j = Point(
            [env.flights[j].position.x + dx_j * dt_step, env.flights[j].position.y + dy_j * dt_step])

        if env.airspace.polygon.contains(position_i) and env.airspace.polygon.contains(position_j):
            if position_i.distance(position_j) < env.alert_distance:
                WillBeConflict = True

    return WillBeConflict


def policy_action(observations, memory, env) -> List:
    actions = [0] * env.num_flights
    FirstStepConflict = [False] * env.num_flights
    InConflict = [False] * env.num_flights
    previous_distances, previous_actions = memory.pop()
    current_distances = env.distances_matrix()

    """ Creating a matrix with ALL flights and its closest conflicts ordered by time, and actualizing conflict parameters """
    FlightsInConflictWith = []
    for i in range(env.num_flights):
        num = []
        time = []
        for j in range(env.num_flights):
            if i not in env.done and j not in env.done and i != j:
                tcpa = t_cpa(env, i, j)
                dcpa = d_cpa(env, i, j)
                tcpa_bearing = t_cpa_bearing(env, i, j)
                dcpa_bearing = d_cpa_bearing(env, i, j)
                if (tcpa < 120 and dcpa < env.alert_distance and previous_distances[i, j] > env.alert_distance) or (
                        current_distances[i, j] < env.alert_distance and env.alert_distance < previous_distances[i, j]):
                    InConflict[i] = True
                    FirstStepConflict[i] = True
                    FirstStepConflict[j] = True
                    time.append(tcpa)
                    num.append(j)
                else:
                    if (tcpa_bearing < 120 and dcpa_bearing < env.alert_distance) or current_distances[i, j] < env.alert_distance:
                        InConflict[i] = True
                        time.append(tcpa_bearing)
                        num.append(j)

        tcpa_sorted = [x for _, x in sorted(zip(time, num))]
        FlightsInConflictWith.append(tcpa_sorted)

    """ For each flight i, the conflict between i and its closest flight j is solved modifying the action i """
    for i in range(env.num_flights):
        if i not in env.done:

            list_i = FlightsInConflictWith[i]
            if not InConflict[i]:
                """ NO CONFLICT """
                actions[i] = env.flights[i].drift

            if InConflict[i]:
                """ CONFLICT """
                if not FirstStepConflict:
                    """ Solving the conflict, in process """
                    actions[i] = previous_actions[i]

                if FirstStepConflict[i]:
                    """ First step in the conflict """

                    """ Solving the most important conflict: The closest conflict in terms of time (tcpa) """
                    j = list_i[0]
                    angle_safe_turn = safe_turn_angle(env, i, j)

                    if env.airspace.polygon.contains(env.flights[i].position) and env.airspace.polygon.contains(
                            env.flights[j].position):

                        """ Computing the angle formed by the position of the intruder flight in respect of it's own track """
                        # -------------------------------------------------------------------
                        """ Respect to flight: i """
                        dx_i = env.flights[j].position.x - env.flights[i].position.x
                        dy_i = env.flights[j].position.y - env.flights[i].position.y
                        compass_i = math.atan2(dx_i, dy_i)
                        compass_i = (compass_i + u.circle) % u.circle

                        angle_i = compass_i - env.flights[i].track

                        if angle_i > math.pi:
                            angle_i = -(u.circle - angle_i)
                        elif angle_i < -math.pi:
                            angle_i = u.circle + angle_i
                        # -------------------------------------------------------------------
                        """ Respect to flight: j """
                        dx_j = env.flights[i].position.x - env.flights[j].position.x
                        dy_j = env.flights[i].position.y - env.flights[j].position.y
                        compass_j = math.atan2(dx_j, dy_j)
                        compass_j = (compass_j + u.circle) % u.circle

                        angle_j = compass_j - env.flights[j].track

                        if angle_j > math.pi:
                            angle_j = -(u.circle - angle_j)
                        elif angle_j < -math.pi:
                            angle_j = u.circle + angle_j
                        # -------------------------------------------------------------------

                        """ Looking at the conditions and, in consequence, applying the actions"""
                        approach = u.circle / 24  # 15º
                        converge = (110 / 360) * u.circle  # 110º
                        turn_45grades = u.circle / 8  # 45º

                        ###############
                        # APPROACHING #
                        ###############
                        # When two aircraft are approaching head-on or approximately so and there is danger of
                        # collision, each shall alter its heading to the right.
                        if abs(angle_i) < approach and abs(angle_j) < approach:
                            actions[i] = angle_safe_turn / 2

                        ##############
                        # CONVERGING #
                        ##############
                        # When two aircraft are converging at approximately the same level, the aircraft that has
                        # the other on its right shall give way.
                        elif approach <= abs(angle_i) <= converge or approach <= abs(angle_j) <= converge:

                            if angle_i > 0:
                                if angle_j > 0:
                                    if env.flights[i].airspeed > env.flights[j].airspeed:
                                        actions[i] = angle_safe_turn
                                else:
                                    actions[i] = angle_safe_turn

                            elif angle_i <= 0:
                                if angle_j <= 0:
                                    if env.flights[i].airspeed > env.flights[j].airspeed:
                                        actions[i] = angle_safe_turn

                        ##############
                        # OVERTAKING #
                        ##############
                        # An aircraft that is being overtaken has the right-of-way and the overtaking aircraft
                        # shall keep out of the way of the other aircraft by altering its heading to the right, and no subsequent change in the relative positions of
                        # the two aircraft shall absolve the overtaking aircraft from this obligation until it is entirely past and clear
                        # In all circumstances, the faster flight that is overtaking shall give way
                        elif abs(angle_j) > converge or abs(angle_i) > converge:

                            if env.flights[i].airspeed > env.flights[j].airspeed:
                                actions[i] = angle_safe_turn

                            # if env.flights[i].airspeed < env.flights[j].airspeed:
                            # print('Flight i is overtaking flight j but going slower')
                        # -------------------------------------------------------------------

                    if len(list_i) > 1:
                        """ If there is a multiple conflict, is solved calculating the angle """

                        previous_angle = angle_safe_turn
                        n = 1
                        while n < len(list_i):
                            k = list_i[n]
                            angle_safe_turn_multiple = safe_turn_angle(env, i, k)

                            max_multiple_angle = max(abs(previous_angle), abs(angle_safe_turn_multiple))
                            if max_multiple_angle == abs(previous_angle):
                                max_multiple_angle = previous_angle
                            if max_multiple_angle == abs(angle_safe_turn_multiple):
                                max_multiple_angle = angle_safe_turn_multiple

                            previous_angle = max_multiple_angle
                            n += 1

                        """ The action of i will be the maximum between angles of closest flight and others """
                        actions[i] = max_multiple_angle
                        # -------------------------------------------------------------------

    return






