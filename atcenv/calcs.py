import math
from atcenv.definitions import *


def t_cpa(env, i: int, j: int) -> float:
    """
    Time to get the closest point of approach of a flight i to j
    :return: time to the closest point of approach, done with straight formula
    """
    dx = env.flights[j].position.x - env.flights[i].position.x
    dy = env.flights[j].position.y - env.flights[i].position.y

    """ Computing relative velocity contemplating bearing """
    vrx = (env.flights[j].airspeed * math.sin(env.flights[j].bearing)) - (env.flights[i].airspeed * math.sin(env.flights[i].bearing))
    vry = (env.flights[j].airspeed * math.cos(env.flights[j].bearing)) - (env.flights[i].airspeed * math.cos(env.flights[i].bearing))

    if i == j:
        tcpa = 0
    else:
        tcpa = max(0, -(dx * vrx + dy * vry) / (vrx ** 2 + vry ** 2))

    return tcpa


def d_cpa(env, i: int, j: int) -> float:
    """
    Distance to get the closest point of approach of a flight i to j
    :return: distance to the closest point of approach
    """
    dx = env.flights[j].position.x - env.flights[i].position.x
    dy = env.flights[j].position.y - env.flights[i].position.y
    vrx = env.flights[j].components[0] - env.flights[i].components[0]
    vry = env.flights[j].components[1] - env.flights[i].components[1]
    if i == j:
        dcpa = 0
    else:
        tcpa = t_cpa(env, i, j)
        dcpa = math.sqrt((dx + vrx * tcpa) ** 2 + (dy + vry * tcpa) ** 2)
    return dcpa


def safe_turn_angle(env, i: int, j: int) -> float:
    """
    def function(angle):
        dx = env.flights[j].position.x - env.flights[i].position.x
        dy = env.flights[j].position.y - env.flights[i].position.y
        tcpa = max(0, -(dx * vx[1] + dy * vx[2]) / (vx[1] ** 2 + vx[2] ** 2))
        return [math.sqrt((dx + vx[1] * tcpa) ** 2 + (dy + vx[2] * tcpa) ** 2) - dcpa, math.atan2(vx[1], vx[2]) - angle]
    """
    tol = 100
    angle_right = 0
    while angle_right < (180 * (u.circle / 360)):
        track = env.flights[i].track + angle_right
        dx = env.flights[j].position.x - env.flights[i].position.x
        dy = env.flights[j].position.y - env.flights[i].position.y
        vrx = env.flights[j].components[0] - env.flights[i].airspeed * math.sin(track)
        vry = env.flights[j].components[1] - env.flights[i].airspeed * math.cos(track)
        tcpa = max(0, -(dx * vrx + dy * vry) / (vrx ** 2 + vry ** 2))
        dcpa = math.sqrt((dx + vrx * tcpa) ** 2 + (dy + vry * tcpa) ** 2)

        tol = env.alert_distance - dcpa
        if tol < 0 or tcpa == 0:
            break
        angle_right = angle_right + (1 * (u.circle / 360))

    tol = 100
    angle_left = 0
    while angle_left > (-180 * (u.circle / 360)):
        track = env.flights[i].track + angle_left
        dx = env.flights[j].position.x - env.flights[i].position.x
        dy = env.flights[j].position.y - env.flights[i].position.y
        vrx = env.flights[j].components[0] - env.flights[i].airspeed * math.sin(track)
        vry = env.flights[j].components[1] - env.flights[i].airspeed * math.cos(track)
        tcpa = max(0, -(dx * vrx + dy * vry) / (vrx ** 2 + vry ** 2))
        dcpa = math.sqrt((dx + vrx * tcpa) ** 2 + (dy + vry * tcpa) ** 2)

        tol = env.alert_distance - dcpa
        if tol < 0 or tcpa == 0:
            break
        angle_left = angle_left - (1 * (u.circle / 360))

    """ Choosing whether turn right or turn left """
    if env.flights[i].position.distance(env.flights[j].position) < env.alert_distance:   # Same as tcpa == 0
        # Computing track right and verifying that the angle is between 0 and 2*phi
        track_right = env.flights[i].track + angle_right
        if track_right > 2*u.circle:
            track_right = track_right - u.circle
        # Computing track left and verifying that the angle is between 0 and 2*phi
        track_left = env.flights[i].track + angle_left
        if track_left < 0:
            track_left = track_left + u.circle
        # Computing...
        dif_right = abs(track_right - env.flights[j].track)
        dif_left = abs(track_left - env.flights[j].track)
        if dif_right > dif_left:
            angle_safe_turn = angle_right
        else:
            angle_safe_turn = angle_left
    else:
        angle_safe_turn = min(abs(angle_right), abs(angle_left))
        if angle_safe_turn == abs(angle_left):
            angle_safe_turn = angle_left

    return angle_safe_turn


def sector_assignment(rel_angle):
    converge = (110 / 360) * u.circle  # 110º
    head_on = (15 / 360) * u.circle    # 15º
    overtake = (180 / 360) * u.circle  # 180º
    if rel_angle > 0:
        if rel_angle <= head_on:
            sector = 0
        elif rel_angle > head_on & rel_angle <= converge / 2:
            sector = 1
        elif rel_angle > converge / 2 & rel_angle <= converge:
            sector = 2
        elif rel_angle > converge & rel_angle <= overtake:
            sector = 3
    else:
        rel_angle = abs(rel_angle)
        if rel_angle <= head_on:
            sector = 7
        elif rel_angle > head_on & rel_angle <= converge / 2:
            sector = 6
        elif rel_angle > converge / 2 & rel_angle <= converge:
            sector = 5
        elif rel_angle > converge & rel_angle <= overtake:
            sector = 4
    return sector


def relative_angle(x1, y1, x2, y2, track1):
    rel_angle = convert_angle(((math.atan2(y2 - y1, x2 - x1) + u.circle) % u.circle) - track1)
    return rel_angle


def convert_angle(angle):
    if angle > math.pi:
        angle = -(u.circle - angle)
    elif angle < -math.pi:
        angle = u.circle + angle
    return angle


def approx_angle(angle):
    # approximates angle to have multiples of 10.
    new_angle = math.ceil(angle / 10) * 10
    return new_angle


def action_from_angle(angle, flight, turn_intensity):
    if angle == 0:
        action = 0
    elif angle == flight.bearing - flight.track:
        """TODO: idk si es del reves"""
        action = 1
    else:
        angle = approx_angle(angle)
        action = (angle * 2) / 10
        if action < 0:
            action = abs(action) + 1
    return action
