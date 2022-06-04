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


def policy_action(observations, env) -> List:
    actions = [0] * env.num_flights
    solved = [False] * env.num_flights
    current_distances = env.distances_matrix()

    """ Creating a matrix with ALL flights and its closest conflicts ordered by distance """
    FlightsInConflictWith = []
    for i in range(env.num_flights):
        num = []
        dist = []
        for j in range(env.num_flights):
            if i not in env.done and j not in env.done and i != j:
                tcpa = calcs.t_cpa(env, i, j)
                dcpa = calcs.d_cpa(env, i, j)
                if (tcpa < 120 and dcpa < env.alert_distance) or (current_distances[i, j] < env.alert_distance):
                    dist.append(dcpa)
                    num.append(j)
        dist_sorted = [x for _, x in sorted(zip(dist, num))]
        FlightsInConflictWith.append(dist_sorted)

    """ For each flight i, the conflict between i and its closest flight j is solved modifying the action i """
    for i in range(env.num_flights):
        if i is not env.done:
            list_i = FlightsInConflictWith[i]

            if len(list_i) == 0:
                """ NO CONFLICT """
                actions[i] = 0
                solved[i] = True

            else:
                j = list_i[0]
                if env.airspace.polygon.contains(env.flights[i].position) and env.airspace.polygon.contains(env.flights[j].position):

                    """Computing the angle formed by the position of the intruder flight in respect of its own track """
                    # Respect to flight: i
                    dx_i = env.flights[j].position.x - env.flights[i].position.x
                    dy_i = env.flights[j].position.y - env.flights[i].position.y
                    compass_i = math.atan2(dx_i, dy_i)
                    compass_i = (compass_i + u.circle) % u.circle

                    angle_i = calcs.convert_angle(compass_i - env.flights[i].track)

                    # Respect to flight: j
                    dx_j = - dx_i
                    dy_j = - dy_i
                    compass_j = math.atan2(dx_j, dy_j)
                    compass_j = (compass_j + u.circle) % u.circle

                    angle_j = calcs.convert_angle(compass_j - env.flights[j].track)

                    """ Looking at the conditions and, in consequence, applying the actions"""

                    approach = u.circle / 24  # 15º
                    converge = (110 / 360) * u.circle  # 110º
                    turn_45grades = u.circle / 8  # 45º

                    ###############
                    # APPROACHING #
                    ###############
                    # When two aircraft are approaching head-on or approximately so and there is danger of
                    # collision, each shall alter its heading to the right.
                    if abs(angle_i) <= approach:
                        actions[i] = 1

                    ##############
                    # CONVERGING #
                    ##############
                    # When two aircraft are converging at approximately the same level, the aircraft that has
                    # the other on its right shall give way.
                    elif approach <= abs(angle_i) <= converge:

                        if angle_i > 0:
                            actions[i] = 2

                        elif angle_i <= 0:
                            actions[i] = 0
                            if angle_j <= 0:
                                actions[i] = 1

                    ##############
                    # OVERTAKING #
                    ##############
                    # An aircraft that is being overtaken has the right-of-way and the overtaking aircraft
                    # shall keep out of the way of the other aircraft by altering its heading to the right.
                    # In all circumstances, the faster flight that is overtaking shall give way
                    elif abs(angle_i) > converge:
                        if env.flights[i].airspeed < env.flights[j].airspeed:
                            actions[i] = 0

                    elif abs(angle_j) > converge:
                        if env.flights[i].airspeed > env.flights[j].airspeed:
                            actions[i] = 1
    return actions






