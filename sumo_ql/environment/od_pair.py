from typing import List
import random as rd
from datetime import datetime

class ODPair:
    """Class responsible for holding information regarding OD-pairs that are necessary in the environment.

        Args:
            straight_distance (float): distance between the origin and destination as a straight line.
    """

    def __init__(self, straight_distance: float) -> None:
        self.__straight_distance: float = straight_distance
        self.__min_load: int = -1
        self.__current_load: List[str] = []
        self.__vehicles_within: List[str] = []

        rd.seed(datetime.now())

    @property
    def min_load(self) -> int:
        """Property that returns the OD-pair minimum load required (number of vehicles running between the OD-pair).

        Returns:
            int: Minimum number of vehicles required to be running within the OD-pair
        """
        return self.__min_load

    @min_load.setter
    def min_load(self, val: int) -> None:
        """Setter for the minimum load of the OD-pair.

        Args:
            val (int): value to set the minimum load on the OD-pair

        Raises:
            RuntimeError: the method raises a RuntimeError if the value is negative.
        """
        if val <= 0:
            raise RuntimeError("Value should be positive!")

        self.__min_load = val

    @property
    def curr_load(self) -> int:
        return len(self.__current_load)

    @property
    def has_enough_vehicles(self) -> bool:
        """Property that returns a boolean indicating if the OD-pair has enough vehicles running within the OD-pair.

        Returns:
            bool: value that indicates if the OD-pair's current load is higher or equal than the minimum required.
        """
        return len(self.__current_load) >= self.__min_load

    @property
    def straight_distance(self) -> float:
        """Property that returns the straight line distance between the origin and destination of the OD-pair.

        Returns:
            float: straight distance between origin and destination of the OD-pair.
        """
        return self.__straight_distance

    def increase_load(self, vehicle_id) -> None:
        """Method that increases the OD-pair current load by 1.
        """
        self.__current_load.append(vehicle_id)

    def decrease_load(self, vehicle_id) -> None:
        """Method that decreases the OD-pair current load by 1.
        """
        self.__current_load.remove(vehicle_id)

    def reset(self) -> None:
        """Method that resets the OD-pair (sets the current load to 0).
        """
        self.__current_load = []

    def append_vehicle(self, vehicle_id: str) -> None:
        self.__vehicles_within.append(vehicle_id)
        
    def random_vehicle(self) -> str:
        return rd.choice([vehicle_id for vehicle_id in self.__vehicles_within if vehicle_id not in self.__current_load])
