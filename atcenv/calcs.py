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


def t_cpa_bearing(env, i: int, j: int) -> float:
    """
    Time to get the closest point of approach of a flight i to j following its bearing
    :return: time to the closest point of approach, done with straight formula
    """
    track_i = env.flights[i].track + env.flights[i].drift
    track_j = env.flights[j].track + env.flights[j].drift

    """ Computing relative velocity contemplating bearing """
    dx = env.flights[j].position.x - env.flights[i].position.x
    dy = env.flights[j].position.y - env.flights[i].position.y

    """ Computing relative velocity contemplating track --> x """
    vrx = env.flights[j].airspeed * math.sin(track_j) - env.flights[i].airspeed * math.sin(track_i)
    vry = env.flights[j].airspeed * math.cos(track_j) - env.flights[i].airspeed * math.cos(track_i)

    if i == j:
        tcpa_bearing = 0
    else:
        tcpa_bearing = max(0, -(dx * vrx + dy * vry) / (vrx ** 2 + vry ** 2))

    return tcpa_bearing


def d_cpa_bearing(env, i: int, j: int) -> float:
    """
    Distance to get the closest point of approach of a flight i to j following its bearing
    :return: distance to the closest point of approach
    """
    track_i = env.flights[i].track + env.flights[i].drift
    track_j = env.flights[j].track + env.flights[j].drift
    dx = env.flights[j].position.x - env.flights[i].position.x
    dy = env.flights[j].position.y - env.flights[i].position.y
    vrx = env.flights[j].airspeed * math.sin(track_j) - env.flights[i].airspeed * math.sin(track_i)
    vry = env.flights[j].airspeed * math.cos(track_j) - env.flights[i].airspeed * math.cos(track_i)

    if i == j:
        dcpa_bearing = 0
    else:
        tcpa = t_cpa_bearing(env, i, j)
        dcpa_bearing = math.sqrt((dx + vrx * tcpa) ** 2 + (dy + vry * tcpa) ** 2)

    return dcpa_bearing


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
    converge = 110 * u.circle/360  # 110º
    head_on = 15 * u.circle/360  # 15º
    overtake = math.pi  # 180º
    # print(rel_angle, head_on, converge/2, converge, overtake)
    if rel_angle > overtake:
        angle_diff = rel_angle - overtake
        rel_angle = -math.pi + angle_diff
    if rel_angle < -overtake:
        angle_diff = rel_angle + overtake
        rel_angle = math.pi + angle_diff
    if rel_angle >= 0:
        if rel_angle <= head_on:
            sector = 0
        elif head_on < rel_angle <= converge / 2:
            sector = 1
        elif converge / 2 < rel_angle <= converge:
            sector = 2
        elif converge < rel_angle <= overtake:
            sector = 3
    else:
        rel_angle = abs(rel_angle)
        if rel_angle <= head_on:
            sector = 7
        elif head_on < rel_angle <= converge / 2:
            sector = 6
        elif converge / 2 < rel_angle <= converge:
            sector = 5
        elif converge < rel_angle <= overtake:
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


def approx_angle(angle, angle_change):
    # approximates angle to have multiples of angle_change.
    new_angle = math.ceil(angle / angle_change) * angle_change
    return new_angle


def action_from_angle(angle, flight, angle_change, num_actions):
    if angle == 0:
        action = 0
    elif angle == flight.drift:
        action = 1
    else:
        angle = approx_angle(angle, angle_change)
        action = (angle * 2) // angle_change
        if action < 0:
            action = abs(action) + 1
        if action > num_actions - 1:
            # maximum turn
            action = num_actions - 1
    return action
