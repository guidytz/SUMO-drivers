from __future__ import annotations
from typing import List
from typing import TYPE_CHECKING

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

    def __init__(self, vehicle_id: str,
                 origin: str,
                 destination: str,
                 arrival_bonus: int,
                 wrong_destination_penalty: int,
                 original_route: List[str],
                 environment: SumoEnvironment) -> None:
        self.__id = vehicle_id
        self.__origin = origin
        self.__destination = destination
        self.__arrival_bonus = arrival_bonus
        self.__wrong_destination_penalty = wrong_destination_penalty
        self.__original_route = original_route
        self.__environment = environment
        self.__current_link = None
        self.__load_time = -1.0
        self.__departure_time = -1.0
        self.__arrival_time = -1.0
        self.__last_link_departure_time = -1.0
        self.__travel_time_last_link = -1.0
        self.__route = list([self.__origin])

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
        """'property that returns the vehicle's current route

        Returns:
            List[str]: vehicle's current route
        """
        return self.__route

    def compute_reward(self, use_bonus_or_penalty: bool = True) -> int:
        """Method that computes the reward the agent should receive based on its last action.
        The reward is based on the vehicle's last travel time plus a bonus (if the destination is the vehicle's expected
        destination) or minus a penalty (if the vehicle reaches a destination node that isn't its expected destination)

        Args:
            use_bonus_or_penalty (bool, optional): argument that defines if the vehicle should compute the reward with
            bonus or penalty (in case of the vehicle reaching a destination node). Defaults to True.

        Raises:
            RuntimeError: the method raises a RuntimeError if the vehicle hasn't dearted yet

        Returns:
            int: reward calculated
        """
        if not self.departed:
            raise RuntimeError(f"Vehicle {self.__id}  hasn't departed yet!")
        reward = - self.__travel_time_last_link
        if self.reached_destination and use_bonus_or_penalty:
            if self.__route[-1] != self.__destination:
                reward -= self.__wrong_destination_penalty
            else:
                reward += self.__arrival_bonus
        return reward

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
    def current_link(self) -> str:
        """Property that returns the vehicle's current link

        Raises:
            RuntimeError: the method raises a RuntimeError if the vehicle hasn't departed yet

        Returns:
            str: vehicle's current link id
        """
        if not self.departed:
            raise RuntimeError(f"Vehicle {self.__id} hasn't departed yet!")
        return self.__current_link

    def update_current_link(self, link: str, current_time: int) -> None:
        """Method to update the vehicle's current link.

        Args:
            link (str): new current link's id
            current_time (int): time step the vehile has entered the new link

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
        self.__current_link = link
        self.__append_destination_node()
        self.__last_link_departure_time = current_time

    def __compute_last_link_travel_time(self, current_time: int) -> None:
        """Method that computes the travel time taken in last link traveled using time the vehicle departed in the link
        and the current time.

        Args:
            current_time (int): current time step
        """
        self.__travel_time_last_link = current_time - self.__last_link_departure_time

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

    def is_in_link(self, link: str) -> bool:
        """Method that tests if the vehicle is in the given link.

        Args:
            link (str): link ID to verify.

        Returns:
            bool: the value of the test if the given link is equal to the vehicle's current link.
        """
        return link == self.__current_link

    def __append_destination_node(self) -> None:
        """Method that appends the destination node of the vehicle's current link to its route.
        """
        if self.__current_link is not None:
            destination_node = self.__environment.get_link_destination(self.__current_link)
            self.__route.append(destination_node)
