"""
Environment module
"""
import gym
import collections
from typing import Dict, List
from atcenv.definitions import *
from gym.envs.classic_control import rendering
from shapely.geometry import LineString

from . import *
import numpy as np

from . import policy, calcs

WHITE = [255, 255, 255]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
YELLOW = [255, 255, 0]


class Environment(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self,
                 num_flights: int = 10,
                 dt: float = 5.,
                 max_area: Optional[float] = 200. * 200.,
                 min_area: Optional[float] = 125. * 125.,
                 max_speed: Optional[float] = 500.,
                 min_speed: Optional[float] = 400,
                 max_episode_len: Optional[int] = 300,
                 min_distance: Optional[float] = 5.,
                 alert_distance: Optional[float] = 8.,
                 distance_init_buffer: Optional[float] = 5.,
                 angle_change: Optional[int] = 10,
                 **kwargs):
        """
        Initialises the environment

        :param num_flights: numer of flights in the environment
        :param dt: time step (in seconds)
        :param max_area: maximum area of the sector (in nm^2)
        :param min_area: minimum area of the sector (in nm^2)
        :param max_speed: maximum speed of the flights (in kt)
        :param min_speed: minimum speed of the flights (in kt)
        :param max_episode_len: maximum episode length (in number of steps)
        :param min_distance: pairs of flights which distance is < min_distance are considered in conflict (in nm)
        :param distance_init_buffer: distance factor used when initialising the enviroment to avoid flights close to conflict and close to the target
        :param kwargs: other arguments of your custom environment
        """
        self.num_flights = num_flights
        self.max_area = max_area * (u.nm ** 2)
        self.min_area = min_area * (u.nm ** 2)
        self.max_speed = max_speed * u.kt
        self.min_speed = min_speed * u.kt
        self.min_distance = min_distance * u.nm
        self.alert_distance = alert_distance * u.nm
        self.max_episode_len = max_episode_len
        self.distance_init_buffer = distance_init_buffer
        self.dt = dt

        # tolerance to consider that the target has been reached (in meters)
        self.tol = self.max_speed * 1.05 * self.dt

        self.viewer = None
        self.airspace = None
        self.flights = []  # list of flights
        self.i = None

        # conflict-related
        self.conflicts = set()  # set of flights that are in conflict
        self.alerts = set()     # set of flights that are in alert
        self.done = set()       # set of flights that reached the target
        self.n_conflicts_step = 0
        self.n_conflicts_episode = 0

        # human policy-related
        self.angle_change = angle_change
        self.matrix_real_conflicts_episode = np.full((self.num_flights, self.num_flights), False)

        # reinforcement learning-related
        self.n_neighbours = 2
        self.num_discrete_actions = 3
        self.observation_space = []
        self.action_space = []
        for agent in range(self.num_flights):
            self.observation_space.append(gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=float))
            self.action_space.append(gym.spaces.Box(low=(- u.circle / 2), high=(u.circle / 2), shape=(1,)))

    def resolution(self, action: List) -> None:
        """
        Applies the resolution actions
        If your policy can modify the speed, then remember to clip the speed of each flight
        In the range [min_speed, max_speed]
        :param action: list of resolution actions assigned to each flight
        :return:
        """

        for f, j in zip(self.flights, action):
            k = 1
            if j == 0:
                continue
            elif j == 1:
                f.track = f.bearing
            else:
                # angle change is the change we apply to the track. If it's 10 then the intervals will be spaced 10º,
                # if it's 5º they are gonna be 5º, 10º, 15º...
                if j % 2 != 0:
                    k = -1
                f.track += (j // 2) * self.angle_change * k
        return None

    def reward(self) -> List:
        """
        Returns the reward assigned to each agent
        :return: reward assigned to each agent
        """
        rew_array = []
        wot = 1.  # not in optimal track
        wtd = 0.5  # track deviation
        for i in range(self.num_flights):
            if i not in self.done:
                # not in optimal track
                track_penalty = 0
                if abs(self.flights[i].drift) > 1e-3:
                    t_ei = abs(self.flights[i].drift) / math.pi
                    track_penalty = - wot - wtd * t_ei
                # conflicts
                conflicts_penalty = 0
                for k in range(self.num_flights):
                    if k != i:
                        wc = 100
                        if abs(self.flights[i].position.distance(self.flights[k].position)) >= self.min_distance:
                            wc = 0
                        conflicts_penalty -= wc
                reward = track_penalty + conflicts_penalty
                rew_array.append(reward)
            else:
                rew_array.append(np.NaN)
        rew_without_nan = [x for x in rew_array if np.isnan(x) == False]
        if len(rew_without_nan) != 0:
            rew_array = [np.average(rew_without_nan)] * self.num_flights
        else:
            rew_array = [0] * self.num_flights
        return rew_array

    def relative_obs_parameters(self, i: int, j: int) -> List:
        """
        Returns the relative state of each agent
        :return: state parameters of each agent to each other
        """
        x_agent = self.flights[i].position.x
        y_agent = self.flights[i].position.y
        x_flight = self.flights[j].position.x
        y_flight = self.flights[j].position.y
        v_agent = self.flights[i].airspeed
        v_flight = self.flights[j].airspeed

        rel_distance = ((((x_agent - x_flight) ** 2) + ((y_agent - y_flight) ** 2)) ** 0.5)
        rel_airspeed = v_agent - v_flight
        rel_angle = calcs.relative_angle(x_agent, y_agent, x_flight, y_flight, self.flights[i].track)
        t_cpa = calcs.t_cpa(self, i, j) / 60  # in minutes
        d_cpa = calcs.d_cpa(self, i, j)
        parameters = [t_cpa, d_cpa, rel_distance, rel_angle, rel_airspeed]
        return parameters

    def observation(self) -> List:
        """
        Returns the observation of each agent
        :return: observation of each agent
        """
        obs_array = []
        # Get ID closest flight
        for i in range(self.num_flights):
            if i not in self.done:
                tcpa_list = []
                neighbours_info = np.zeros([self.n_neighbours, 5], dtype=float)
                for j in range(self.num_flights):
                    if i != j and j not in self.done:
                        # Compute t_cpa between flights
                        tcpa_ij = calcs.t_cpa(self, i, j)
                        tcpa_list.append(tcpa_ij)
                idx_neighbours = np.argsort(tcpa_list)[:self.n_neighbours]
            # Concatenate observation from the agent and the hidden features
                for k in range(self.n_neighbours):
                    if len(self.done) == self.num_flights - 1:
                        neighbours_info[k, :] = [np.NaN] * 5
                    elif k < self.num_flights - len(self.done) - 1:
                        neighbours_info[k, :] = self.relative_obs_parameters(i, idx_neighbours[k])
                    else:
                        neighbours_info[k, :] = [np.NaN] * 5
                final_agent_observation = neighbours_info.flatten()
            else:
                final_agent_observation = [np.NaN] * self.observation_space[i].shape[0]
            # Add the final observation for the agent
            obs_array.append(final_agent_observation)
        return np.array(obs_array)

    def update_conflicts(self) -> None:
        """
        Updates the set of flights that are in conflict
        Note: flights that reached the target are not considered
        :return:
        """
        # reset set
        self.conflicts = set()
        self.n_conflicts_step = 0

        for i in range(self.num_flights - 1):
            if i not in self.done:
                for j in range(i + 1, self.num_flights):
                    if j not in self.done:
                        distance = self.flights[i].position.distance(self.flights[j].position)
                        if distance < self.min_distance:
                            self.conflicts.update((i, j))
                            self.n_conflicts_step += 1
                            self.n_conflicts_episode += 1
                            self.matrix_real_conflicts_episode[i, j] = True

    def update_alerts(self) -> None:
        """
        Updates the set of flights that are in alert
        Note: flights that reached the target are not considered
        :return:
        """
        # reset set
        self.alerts = set()

        for i in range(self.num_flights - 1):
            if i not in self.done:
                for j in range(i + 1, self.num_flights):
                    if j not in self.done:
                        distance = self.flights[i].position.distance(self.flights[j].position)
                        if distance < self.alert_distance:
                            self.alerts.update((i, j))

    def update_done(self) -> None:
        """
        Updates the set of flights that reached the target
        :return:
        """
        for i, f in enumerate(self.flights):
            if i not in self.done:
                if not self.airspace.polygon.contains(f.position):
                    self.done.add(i)  # stop when out of sector!!
                else:
                    distance = f.position.distance(f.target)
                    if distance < self.tol:
                        self.done.add(i)  # stop when reaching the target

    def update_positions(self) -> None:
        """
        Updates the position of the agents
        Note: the position of agents that reached the target is not modified
        :return:
        """
        for i, f in enumerate(self.flights):
            if i not in self.done:
                # get current speed components
                dx, dy = f.components

                # get current position
                position = f.position

                # get new position and advance one time step
                f.position._set_coords(position.x + dx * self.dt, position.y + dy * self.dt)

                # compute actual trajectory length
                f.actual_distance += ((dx * self.dt) ** 2 + (dy * self.dt) ** 2) ** 0.5

    def step(self, action: List) -> Tuple[List, List, bool]:
        """
        Performs a simulation step

        :param action: list of resolution actions assigned to each flight
        :return: observation, reward, done status and other information
        """

        # apply resolution actions
        self.resolution(action)

        # update positions
        self.update_positions()

        # update done set
        self.update_done()

        # update conflict set
        self.update_conflicts()

        # update alert set
        self.update_alerts()

        # compute reward
        rew = self.reward()

        # compute observation
        obs = self.observation()

        # increase steps counter
        self.i += 1

        # check termination status
        # termination happens when
        # (1) all flights reached the target
        # (2) the maximum episode length is reached
        done = (len(self.done) == self.num_flights or self.i > self.max_episode_len)

        return rew, obs, done

    def reset(self) -> List:
        """
        Resets the environment and returns initial observation
        :return: initial observation
        """
        # create random airspace
        self.airspace = Airspace.random(self.min_area, self.max_area)

        # create random flights
        self.flights = []
        tol = self.distance_init_buffer * self.tol
        min_distance = self.distance_init_buffer * self.min_distance
        while len(self.flights) < self.num_flights:
            valid = True
            candidate = Flight.random(self.airspace, self.min_speed, self.max_speed, tol)

            # ensure that candidate is not in conflict
            for f in self.flights:
                if candidate.position.distance(f.position) < min_distance:
                    valid = False
                    break
            if valid:
                self.flights.append(candidate)

        # initialise steps counter
        self.i = 0

        # clean conflicts and done sets
        self.conflicts = set()
        self.done = set()
        self.n_conflicts_step = 0
        self.n_conflicts_episode = 0
        self.matrix_real_conflicts_episode = np.full((self.num_flights, self.num_flights), False)
        # return initial observation
        return self.observation()

    def render(self, mode=None) -> None:
        """
        Renders the environment
        :param mode: rendering mode
        :return:
        """
        if self.viewer is None:
            # initialise viewer
            screen_width, screen_height = 600, 600

            minx, miny, maxx, maxy = self.airspace.polygon.buffer(10 * u.nm).bounds
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(minx, maxx, miny, maxy)

            # fill background
            background = rendering.make_polygon([(minx, miny),
                                                 (minx, maxy),
                                                 (maxx, maxy),
                                                 (maxx, miny)],
                                                filled=True)
            background.set_color(*BLACK)
            self.viewer.add_geom(background)

            # display airspace
            sector = rendering.make_polygon(self.airspace.polygon.boundary.coords, filled=False)
            sector.set_linewidth(1)
            sector.set_color(*WHITE)
            self.viewer.add_geom(sector)

        # add current positions
        for i, f in enumerate(self.flights):
            if i in self.done:
                continue

            if i in self.conflicts:
                color = RED
            elif i in self.alerts:
                color = YELLOW
            else:
                color = BLUE

            circle = rendering.make_circle(radius=self.min_distance / 2.0,
                                           res=10,
                                           filled=False)
            circle.add_attr(rendering.Transform(translation=(f.position.x,
                                                            f.position.y)))
            circle.set_color(*BLUE)

            alert_zone_circle = rendering.make_circle(radius=self.alert_distance / 2.0,
                                                      res=10,
                                                      filled=False)
            alert_zone_circle.add_attr(rendering.Transform(translation=(f.position.x,
                                                                        f.position.y)))
            alert_zone_circle.set_color(*YELLOW)

            plan = LineString([f.position, f.target])
            self.viewer.draw_polyline(plan.coords, linewidth=1, color=color)
            prediction = LineString([f.position, f.prediction])
            self.viewer.draw_polyline(prediction.coords, linewidth=4, color=color)

            self.viewer.add_onetime(circle)
            self.viewer.add_onetime(alert_zone_circle)

            # ZERO
            if i == 0:
                zero = LineString(
                    [(f.position.x - 1000, f.position.y - 1000), (f.position.x - 6000, f.position.y - 1000),
                     (f.position.x - 6000, f.position.y - 6000), (f.position.x - 6000, f.position.y - 11000),
                     (f.position.x - 1000, f.position.y - 11000), (f.position.x - 1000, f.position.y - 1000)])
                self.viewer.draw_polyline(zero.coords, linewidth=1, color=WHITE)

            # ONE
            if i == 1:
                one = LineString([
                    (f.position.x - 1000, f.position.y - 1000), (f.position.x - 1000, f.position.y - 11000)])
                self.viewer.draw_polyline(one.coords, linewidth=1, color=WHITE)

            # TWO
            if i == 2:
                two = LineString(
                    [(f.position.x - 6000, f.position.y - 1000), (f.position.x - 1000, f.position.y - 1000),
                     (f.position.x - 1000, f.position.y - 6000), (f.position.x - 6000, f.position.y - 6000),
                     (f.position.x - 6000, f.position.y - 11000), (f.position.x - 1000, f.position.y - 11000)])
                self.viewer.draw_polyline(two.coords, linewidth=1, color=WHITE)

            # THREE
            if i == 3:
                three = LineString(
                    [(f.position.x - 6000, f.position.y - 1000), (f.position.x - 1000, f.position.y - 1000),
                     (f.position.x - 1000, f.position.y - 6000), (f.position.x - 6000, f.position.y - 6000),
                     (f.position.x - 1000, f.position.y - 6000), (f.position.x - 1000, f.position.y - 11000),
                     (f.position.x - 6000, f.position.y - 11000)])
                self.viewer.draw_polyline(three.coords, linewidth=1, color=WHITE)

            # FOUR
            if i == 4:
                four = LineString(
                    [(f.position.x - 6000, f.position.y - 1000), (f.position.x - 6000, f.position.y - 6000),
                     (f.position.x - 1000, f.position.y - 6000), (f.position.x - 1000, f.position.y - 1000),
                     (f.position.x - 1000, f.position.y - 6000), (f.position.x - 1000, f.position.y - 11000)])
                self.viewer.draw_polyline(four.coords, linewidth=1, color=WHITE)

            # FIVE
            if i == 5:
                five = LineString(
                    [(f.position.x - 1000, f.position.y - 1000), (f.position.x - 6000, f.position.y - 1000),
                     (f.position.x - 6000, f.position.y - 6000), (f.position.x - 1000, f.position.y - 6000),
                     (f.position.x - 1000, f.position.y - 11000), (f.position.x - 6000, f.position.y - 11000)])
                self.viewer.draw_polyline(five.coords, linewidth=1, color=WHITE)

            # SIX
            if i == 6:
                five = LineString(
                    [(f.position.x - 1000, f.position.y - 1000), (f.position.x - 6000, f.position.y - 1000),
                     (f.position.x - 6000, f.position.y - 6000), (f.position.x - 6000, f.position.y - 11000),
                     (f.position.x - 1000, f.position.y - 11000), (f.position.x - 1000, f.position.y - 6000),
                     (f.position.x - 6000, f.position.y - 6000)])
                self.viewer.draw_polyline(five.coords, linewidth=1, color=WHITE)

            # SEVEN
            if i == 7:
                seven = LineString([
                    (f.position.x - 6000, f.position.y - 1000), (f.position.x - 1000, f.position.y - 1000),
                    (f.position.x - 1000, f.position.y - 11000)])
                self.viewer.draw_polyline(seven.coords, linewidth=1, color=WHITE)

            # EIGHT
            if i == 8:
                eight = LineString(
                    [(f.position.x - 1000, f.position.y - 1000), (f.position.x - 6000, f.position.y - 1000),
                     (f.position.x - 6000, f.position.y - 6000), (f.position.x - 6000, f.position.y - 11000),
                     (f.position.x - 1000, f.position.y - 11000), (f.position.x - 1000, f.position.y - 6000),
                     (f.position.x - 6000, f.position.y - 6000), (f.position.x - 1000, f.position.y - 6000),
                     (f.position.x - 1000, f.position.y - 1000)])
                self.viewer.draw_polyline(eight.coords, linewidth=1, color=WHITE)

            # NINE
            if i == 9:
                nine = LineString(
                    [(f.position.x - 1000, f.position.y - 11000), (f.position.x - 1000, f.position.y - 1000),
                     (f.position.x - 6000, f.position.y - 1000), (f.position.x - 6000, f.position.y - 6000),
                     (f.position.x - 1000, f.position.y - 6000)])
                self.viewer.draw_polyline(nine.coords, linewidth=1, color=WHITE)

        self.viewer.render()

    def close(self) -> None:
        """
        Closes the viewer
        :return:
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def distances_matrix(self):
        dist = np.zeros([self.num_flights, self.num_flights])
        for i in range(self.num_flights):
            for j in range(self.num_flights):
                if i not in self.done and j not in self.done and i != j:
                    distance_to_flight = self.flights[i].position.distance(self.flights[j].position)
                    dist[i, j] = distance_to_flight
        return dist

