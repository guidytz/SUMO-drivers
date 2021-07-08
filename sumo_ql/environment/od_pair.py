class ODPair:
    """Class responsible for holding information regarding OD-pairs that are necessary in the environment.

        Args:
            straight_distance (float): distance between the origin and destination as a straight line.
    """

    def __init__(self, straight_distance: float) -> None:
        self.__straight_distance = straight_distance
        self.__min_load = -1
        self.__current_load = 0

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
    def has_enough_vehicles(self) -> bool:
        """Property that returns a boolean indicating if the OD-pair has enough vehicles running within the OD-pair.

        Returns:
            bool: value that indicates if the OD-pair's current load is higher or equal than the minimum required.
        """
        return self.__current_load >= self.__min_load

    @property
    def straight_distance(self) -> float:
        """Property that returns the straight line distance between the origin and destination of the OD-pair.

        Returns:
            float: straight distance between origin and destination of the OD-pair.
        """
        return self.__straight_distance

    def increase_load(self) -> None:
        """Method that increases the OD-pair current load by 1.
        """
        self.__current_load += 1

    def decrease_load(self) -> None:
        """Method that decreases the OD-pair current load by 1.
        """
        self.__current_load -= 1

    def reset(self) -> None:
        """Method that resets the OD-pair (sets the current load to 0).
        """
        self.__current_load = 0
