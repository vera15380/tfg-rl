"""
Policy module
"""

from typing import List
from shapely.geometry import Point
from atcenv import calcs
from atcenv.definitions import *


def policy_action(memory, env) -> List:
    actions = [0] * env.num_flights
    FirstStepConflict = [False] * env.num_flights
    InConflict = [False] * env.num_flights
    previous_distances, previous_actions = memory.pop()
    current_distances = env.distances_matrix()

    FlightsInConflictWith = calcs.alert_detection(env, previous_distances, previous_actions, current_distances,
                                                  FirstStepConflict, InConflict)

    """ For each flight i, the conflict between i and its closest flight j is solved modifying the action i """
    for i in range(env.num_flights):
        if i not in env.done:

            list_i = FlightsInConflictWith[i]
            if not InConflict[i]:
                """ NO CONFLICT """
                actions[i] = calcs.action_from_angle(env.flights[i].drift, env.flights[i], env.angle_change,
                                                     env.num_discrete_actions)

            if InConflict[i]:
                """ CONFLICT """
                if not FirstStepConflict[i]:
                    """ Solving the conflict, in process """
                    actions[i] = calcs.action_from_angle(0, env.flights[i], env.angle_change, env.num_discrete_actions)

                if FirstStepConflict[i]:
                    """ First step in the conflict """

                    """ Solving the most important conflict: The closest conflict in terms of time (tcpa) """
                    j = list_i[0]
                    angle_safe_turn = calcs.action_from_angle(calcs.safe_turn_angle(env, i, j), env.flights[i],
                                                              env.angle_change, env.num_discrete_actions)

                    if env.airspace.polygon.contains(env.flights[i].position) and env.airspace.polygon.contains(
                            env.flights[j].position):

                        """ Computing the angle formed by the position of the intruder flight in respect of it's own track """
                        angle_i, angle_j = calcs.position_angles(env, i, j)

                        """ Applying the SERA Rules """
                        calcs.SERA_rules_application(env, i, j, angle_i, angle_j, actions, angle_safe_turn)

                    if len(list_i) > 1:
                        """ If there is a multiple conflict, is solved calculating the maximum angle that solves all """

                        actions[i] = calcs.action_from_angle(calcs.safe_turn_multiple_angle(env, angle_safe_turn, list_i, i),
                                                             env.flights[i], env.angle_change, env.num_discrete_actions)

    return actions







