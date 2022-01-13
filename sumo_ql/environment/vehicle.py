from __future__ import annotations
from typing import List, Dict
from typing import TYPE_CHECKING
from collections import defaultdict
import numpy as np
import pandas as pd
import traci
import traci.constants as tc
from traci.exceptions import TraCIException
from sklearn.preprocessing import MaxAbsScaler as scaler

if TYPE_CHECKING:
    from sumo_ql.environment.sumo_environment import SumoEnvironment


class Vehicle:
    """Class responsible for handling vehicles information regarding position
    within the network, current route, travel time and link travel time to
    compute rewards.

    Args:
        id (str): vehicle id (same as used in SUMO's route files)
        origin (str): id of its origin node
        destination (str): id of its destination node
        arrival_bonus (int): bonus to sum to reward if vehicle arrives at the right destination
        wrong_destination_penalty (int): penalty to subtract to reward if vehicle arrives at wrong destination
        environment (SumoEnvironment): objet of the environment the vehicle is within
    """
    _normalizer = None

    def __init__(self, vehicle_id: str,
                 origin: str,
                 destination: str,
                 arrival_bonus: int,
                 wrong_destination_penalty: int,
                 original_route: List[str],
                 environment: SumoEnvironment,
                 objectives: Objectives) -> None:
        self.__id = vehicle_id
        self.__origin = origin
        self.__destination = destination
        self.__arrival_bonus = arrival_bonus
        self.__wrong_destination_penalty = -wrong_destination_penalty
        self.__original_route = original_route
        self.__environment = environment
        self.__current_link = None
        self.__last_link = None
        self.__load_time = -1.0
        self.__just_changed = False
        self.__departure_time = -1.0
        self.__arrival_time = -1.0
        self.__last_link_departure_time = -1.0
        self.__travel_time_last_link = -1.0
        self.__route = list([self.__origin])
        self.__emission = defaultdict(lambda: 0)
        self.__lst_em_rewards = defaultdict(lambda: 0)
        self.__cumulative_em = defaultdict(lambda: 0)
        self.__objectives = objectives
        self.__link_inclusion = [tc.VAR_ROAD_ID] if tc.VAR_ROAD_ID not in self.__objectives.known_objectives else []
        self.__color = None

    def reset(self) -> None:
        """Method that resets important attributes to the vehicle

        Attributes:
            current_link (str): id of the vehicle's current link within the network
            load_time (int): time step when the vehicle is loaded in the simulation
            departure_time (int): time step when the vehicle is actually inserted within the network
            arrival_time (int): time step when the vehicle reaches a destination node and leaves the network
            last_link_departure_time (int): time step when the vehicle entered the last link recorded
            travel_time_last_link (int): time step the vehicle took to travel last link recorded
            route (list(str)): list of ids of all nodes visited by the vehicle
            ready_to_at (bool): variable indicating if the learning agent controling the vehicle can take its next
            action
        """
        self.__current_link = None
        self.__load_time = -1.0
        self.__departure_time = -1.0
        self.__arrival_time = -1.0
        self.__last_link_departure_time = -1.0
        self.__travel_time_last_link = -1.0
        self.__route = list([self.__origin])
        self.__emission = defaultdict(lambda: 0)
        self.__lst_em_rewards = defaultdict(lambda: 0)
        self.__cumulative_em = defaultdict(lambda: 0)

    @property
    def vehicle_id(self) -> str:
        """property that returns its id

        Returns:
            str: vehicle's id
        """
        return self.__id

    @property
    def origin(self) -> str:
        """property that returns its origin

        Returns:
            str: vehicle's origin node id
        """
        return self.__origin

    @property
    def destination(self) -> str:
        """property that returns its destination

        Returns:
            str: vehicle's destination node id
        """
        return self.__destination

    @property
    def od_pair(self) -> str:
        """property that returns its origin-destination pair

        Returns:
            str: vehicle's origin-destination pair separated by a '|'
        """
        return f"{self.__origin}|{self.__destination}"

    @property
    def original_route(self) -> List[str]:
        """Property that returns the vehicle's original route defined in the route file. This route uses links instead
        of nodes.

        Returns:
            List(str): list containing all the links present in the vehicle's original route
        """
        return self.__original_route

    @property
    def load_time(self) -> int:
        """property that returns its load time

        Returns:
            int: vehicle's load time
        """
        return self.__load_time

    @load_time.setter
    def load_time(self, current_time: int) -> None:
        """vehicle's load time setter

        Args:
            current_time (int): current step time to set the load time

        Raises:
            RuntimeError: if the time passed to the method is negative, the method raises a RuntimeError
        """
        if current_time < 0.0:
            raise RuntimeError("Time cannot be negative!")
        self.__load_time = current_time

    @property
    def route(self) -> List[str]:
        """property that returns the vehicle's current route

        Returns:
            List[str]: vehicle's current route
        """
        return self.__route

    @property
    def current_link(self) -> str:
        """property that returns the vehicle's current link

        Returns:
            str: link ID the vehicle is currently in
        """
        return self.__current_link

    @property
    def last_link(self) -> str:
        """property that returns the link the vehicle last passed through before the current one

        Returns:
            str: last link ID the vehicle passed through before the current one
        """
        return self.__last_link

    def compute_reward(self, use_bonus_or_penalty: bool = True, normalize: bool = False) -> np.array:
        """Method that computes the reward the agent should receive based on its last action.
        The reward is based on the vehicle's last travel time plus a bonus (if the destination is the vehicle's expected
        destination) or minus a penalty (if the vehicle reaches a destination node that isn't its expected destination)

        Args:
            use_bonus_or_penalty (bool, optional): argument that defines if the vehicle should compute the reward with
            bonus or penalty (in case of the vehicle reaching a destination node). Defaults to True.

        Raises:
            RuntimeError: the method raises a RuntimeError if the vehicle hasn't dearted yet

        Returns:
            np.array: reward calculated
        """
        reward = list()
        bonus_or_penalty = 0
        if not self.departed:
            raise RuntimeError(f"Vehicle {self.__id}  hasn't departed yet!")
        if self.__objectives.is_valid(tc.VAR_ROAD_ID):
            reward.append(- self.__travel_time_last_link)

        for key in self.__emission:
            if self.__objectives.is_valid(key):
                em_sum = self.__lst_em_rewards[key]
                reward.append(- em_sum)
        if self.reached_destination and use_bonus_or_penalty:
            if self.__route[-1] != self.__destination:
                bonus_or_penalty = self.__wrong_destination_penalty
            else:
                bonus_or_penalty = self.__arrival_bonus

        norm_reward = self.normalizer.transform([np.array(reward)])[0] if normalize else np.array(reward)
        reward = (lambda val: val + bonus_or_penalty)(norm_reward)
        return reward

    @property
    def normalizer(self) -> scaler:
        """Property that returns the normalizer used to fit the reward data.

        Raises:
            RuntimeError: it raises a RuntimeError if the data file that contains fit data does not exist.

        Returns:
            scaler: the normalizer that can be used to normalize reward data.
        """
        if type(self)._normalizer is None:
            path = self.__environment.sim_path
            fit_file = f"{path}/fit_data_{'_'.join(self.__objectives.objectives_str_list)}.csv"
            try:
                fit_data = pd.read_csv(fit_file).to_numpy()
                type(self)._normalizer = scaler().fit(fit_data)
            except FileNotFoundError:
                err_str = "Fit data must be in scenario directory."
                err_str += " Please run simulation with flag '--collect' before"
                raise RuntimeError(err_str) from FileNotFoundError
        return type(self)._normalizer

    @property
    def ready_to_act(self) -> bool:
        """property that returns the attribute indicating if the learning agent controlling the vehicle can take its
        next action

        Returns:
            bool: attribute indicating if the vehicle is ready to act
        """
        if self.__current_link is None:
            return False

        destination_node = self.__environment.get_link_destination(self.__current_link)
        return destination_node != self.__destination and not self.__environment.is_border_node(destination_node)

    @property
    def is_correct_arrival(self) -> bool:
        """Property that returns a boolean stating if the vehicle arrived at the right destination.

        Returns:
            bool: attribute indicating if the vehicle arrived at the right destination.
        """
        return self.route[-1] == self.destination

    def insert(self) -> bool:
        """Method that inserts vehicle within the network using the original route

        Returns:
            bool: boolean that indicates if the insertion was successful
        """
        route_id = f"r_{self.vehicle_id}"
        inserted = True
        try:
            traci.vehicle.add(self.vehicle_id, route_id)
        except TraCIException:
            print(f"Warning: tried to insert vehicle {self.vehicle_id} to a non existent route. Please verify.")
            traci.route.add(route_id, self.original_route)
            traci.vehicle.add(self.vehicle_id, route_id)
            inserted = False
        traci.vehicle.setColor(self.vehicle_id, self.__color)

        return inserted

    def update_route(self, action: int) -> None:
        """Method that updates the vehicles current route, given an action (link) chosen

        Args:
            action (int): action that determines the link chosen to go through
        """
        try:
            node_id = self.__environment.get_link_destination(self.current_link)
            next_link_id = self.__environment.get_action_link(node_id, action)
            current_route = [self.current_link, next_link_id]
        except RuntimeError:
            node_id = self.origin
            next_link_id = self.__environment.get_action_link(node_id, action)
            current_route = [next_link_id]

        try:
            traci.vehicle.setRoute(self.vehicle_id, current_route)
        except TraCIException:
            destination = self.__environment.get_link_destination(next_link_id)
            print(f"Warning: could not set next link for vehicle ID {self.vehicle_id}.")
            print(f"Current route: {self.route}.")
            print(f"Tried to choose link {next_link_id} to reach node {destination}.")
            print(f"{traci.vehicle.getRoadID(self.vehicle_id) = }")

    def update_data(self, current_time: int) -> None:
        """Method that performs an update in all vehicle's data like link updates and emission data

        Args:
            current_time (int): current simulation step
        """
        traci_vehicle_info = traci.vehicle.getSubscriptionResults(self.vehicle_id)
        self.__last_link = self.current_link
        if (current_link := traci_vehicle_info.pop(tc.VAR_ROAD_ID, self.current_link)) != self.current_link:
            if self.__environment.is_link(current_link):
                self.__last_link = self.__update_current_link(current_link, current_time)
                self.__just_changed = True
        self.__update_emission(traci_vehicle_info)

    @property
    def changed_link(self) -> bool:
        """property that indicates if the vehicle has just changed to a new link

        Returns:
            bool: boolean that indicates if the vehicle has changed to a new link
        """
        if (changed := self.__just_changed):
            self.__just_changed = False
        return changed

    def departure(self, current_time: int) -> None:
        """Method that updates vehicle data when it just entered the network

        Args:
            current_time (int): current simulation step
        """
        self.__update_current_link(traci.vehicle.getRoadID(self.vehicle_id), current_time)
        traci.vehicle.subscribe(self.vehicle_id, self.__link_inclusion + self.__objectives.known_objectives)
        if self.__color is None:
            self.__color = traci.vehicle.getColor(self.vehicle_id)

    def set_arrival(self, current_time: int) -> None:
        """Method that sets the arrival time according to the current time step given.

        Args:
            current_time (int): current time step

        Raises:
            RuntimeError: the method raises a RuntimeError if the vehicle hasn't departed yet
            RuntimeError: the method raises a RuntimeError if the time given is lower than departure time
        """
        if not self.departed:
            raise RuntimeError(f"Vehicle {self.__id} hasn't even departed yet!")
        if current_time < self.__departure_time:
            raise RuntimeError("Invalid arrival time: value lower than departure time!")

        self.__arrival_time = current_time
        self.__update_emission({key: 0 for key in self.__emission})

    @property
    def reached_destination(self) -> bool:
        """Property that indicates if the vehicle has reached a destination node.

        Returns:
            bool: boolean value indicating if vehicle has reached a destination node
        """
        return self.__arrival_time != -1

    @property
    def departed(self) -> bool:
        """Property that indicates if the vehicle has been inserted in the network.

        Returns:
            bool: boolean value indicating if the vehicle has departed in the network.
        """
        return self.__departure_time != -1

    @property
    def travel_time(self) -> int:
        """Property that return the vehicle's travel time if it traveled until a destination node.

        Raises:
            RuntimeError: the method raises a RuntimeError if the vehicle hasn't departed yet
            RuntimeError: the method raises a RuntimeError if the vehicle hasn't reached a destination yet

        Returns:
            int: vehicle's route travel time
        """
        if not self.departed:
            raise RuntimeError("Vehicle hasn't departed yet!")
        if not self.reached_destination:
            raise RuntimeError("Vehicle hasn't reached destination yet!")

        return self.__arrival_time - self.__departure_time

    @property
    def cumulative_data(self) -> List[int]:
        return [self.travel_time] + list(self.__cumulative_em.values())

    def is_in_link(self, link: str) -> bool:
        """Method that tests if the vehicle is in the given link.

        Args:
            link (str): link ID to verify.

        Returns:
            bool: the value of the test if the given link is equal to the vehicle's current link.
        """
        return link == self.__current_link

    def __update_current_link(self, link: str, current_time: int) -> None:
        """Method to update the vehicle's current link.

        Args:
            link (str): new current link's id
            current_time (int): time step the vehicle has entered the new link

        Raises:
            RuntimeError: the method raises a RuntimeError if the vehicle hasn't departed yet and the time given is
            lower than the load time
        """
        if not self.departed:
            if current_time < self.__load_time:
                raise RuntimeError("Invalid departure time: value lower than load time!")
            self.__departure_time = current_time
        else:
            self.__compute_last_link_travel_time(current_time)
        last_link = self.__current_link
        self.__current_link = link
        self.__append_destination_node()
        self.__last_link_departure_time = current_time

        return last_link

    def __update_emission(self, consumption_data: dict) -> None:
        """Method that updates emission data given a dictionary with emission for each type

        Args:
            consumption_data (dict): dictionary containing emission for each type available in simulation
        """
        if self.__just_changed or self.reached_destination:
            for key in self.__emission:
                self.__cumulative_em[key] += self.__emission[key]
                self.__lst_em_rewards[key] = self.__emission[key]
            self.__emission = defaultdict(lambda: 0)
        for key, value in consumption_data.items():
            self.__emission[key] += value

    def __compute_last_link_travel_time(self, current_time: int) -> None:
        """Method that computes the travel time taken in last link traveled using time the vehicle departed in the link
        and the current time.

        Args:
            current_time (int): current time step
        """
        self.__travel_time_last_link = current_time - self.__last_link_departure_time

    def __append_destination_node(self) -> None:
        """Method that appends the destination node of the vehicle's current link to its route.
        """
        if self.__current_link is not None:
            destination_node = self.__environment.get_link_destination(self.__current_link)
            self.__route.append(destination_node)


class Objectives:
    """Class that holds objective params for Multi-objective learning
    """
    __conversions: Dict[str, int] = {
                "TravelTime": tc.VAR_ROAD_ID,
                "CO": tc.VAR_COEMISSION,
                "CO2": tc.VAR_CO2EMISSION,
                "HC": tc.VAR_HCEMISSION,
                "PMx": tc.VAR_PMXEMISSION,
                "NOx": tc.VAR_NOXEMISSION,
                "Fuel": tc.VAR_FUELCONSUMPTION
            }

    def __init__(self, params) -> None:
        self.__known_objectives: List[int] = Objectives.__retrieve_objectives(params)

    @property
    def known_objectives(self) -> List[int]:
        """Known objectives for the current simulation

        Returns:
            List[int]: list containing all the objetives for the given simulation
        """
        return self.__known_objectives

    @property
    def objectives_str_list(self) -> List[str]:
        """Property that returns a list containing all objective strings for the current simulation's objectives.

        Returns:
            List[str]: list containing all objective strings.
        """
        return [string for string, value in self.__conversions.items() if value in self.known_objectives]

    def is_valid(self, objective: int) -> bool:
        """Method that returns a boolean that indicates if a given param is a valid objective

        Args:
            objective (int): param to test if it's a valid objetive

        Returns:
            bool: boolean indicating if the param given is a valid objective
        """
        return objective in self.__known_objectives

    @classmethod
    def __retrieve_objectives(cls, params: List[str]) -> List[int]:
        """Function that converts a list of string objetives to their respective IDs

        Args:
            params (List[str]): list of objective strings to convert to IDs

        Returns:
            List[int]: list of valid objective IDs for the given string listt
        """

        return list(filter(lambda x: x is not None, [cls.__conversions.get(par) for par in params]))
