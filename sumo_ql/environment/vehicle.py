class Vehicle:
    """Class responsible for handling vehicles information regarding position
    within the network, current route, travel time and link travel time to 
    compute rewards.
    """

    def __init__(self, id, origin, destination, arrival_bonus, wrong_destination_penalty, environment):
        """Class constructor

        Args:
            id (str): vehicle id (same as used in SUMO's route files)
            origin (str): id of its origin node
            destination (str): id of its destination node
            arrival_bonus (int): bonus to sum to reward if vehicle arrives at the right destination
            wrong_destination_penalty (int): penalty to subtract to reward if vehicle arrives at wrong destination
            environment (SumoEnvironment): objet of the environment the vehicle is within
        """
        self.__id = id
        self.__origin = origin
        self.__destination = destination
        self.__arrival_bonus = arrival_bonus
        self.__wrong_destination_penalty = wrong_destination_penalty
        self.__environment = environment
        self.reset()

    def reset(self):
        """Method that resets important attributes to the vehicle

        Attributes:
            current_link (str): id of the vehicle's current link within the network
            load_time (int): time step when the vehicle is loaded in the simulation
            departure_time (int): time step when the vehicle is actually inserted within the network
            arrival_time (int): time step when the vehicle reaches a destination node and leaves the network
            last_link_departure_time (int): time step when the vehicle entered the last link recorded
            travel_time_last_link (int): time step the vehicle took to travel last link recorded
            route (list(str)): list of ids of all nodes visited by the vehicle
            ready_to_at (bool): variable indicating if the learning agent controling the vehicle can take its next action
        """
        self.__current_link = None
        self.__load_time = -1.0
        self.__departure_time = -1.0
        self.__arrival_time = -1.0
        self.__last_link_departure_time = -1.0
        self.__travel_time_last_link = -1.0
        self.__route = list([self.__origin])
        self.__ready_to_act = False

    @property
    def id(self):
        """property that returns its id

        Returns:
            str: vehicle's id
        """
        return self.__id

    @property
    def origin(self):
        """property that returns its origin

        Returns:
            str: vehicle's origin node id
        """
        return self.__origin

    @property
    def destination(self):
        """property that returns its destination

        Returns:
            str: vehicle's destination node id
        """
        return self.__destination
    
    @property
    def od_pair(self):
        """property that returns its origin-destination pair    

        Returns:
            str: vehicle's origin-destination pair separated by a '|'
        """
        return f"{self.__origin}|{self.__destination}"


    @property
    def load_time(self):
        """property that returns its load time

        Returns:
            int: vehicle's load time
        """
        return self.__load_time

    @load_time.setter
    def load_time(self, current_time):
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
    def route(self):
        """'property that returns the vehicle's current route

        Returns:
            list(str): vehicle's current route
        """
        return self.__route

    def compute_reward(self, use_bonus_or_penalty=True):
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
            raise RuntimeError("Vehicle hasn't departed yet!")
        reward = - self.__travel_time_last_link
        if self.reached_destination and use_bonus_or_penalty:
            if self.__route[-1] != self.__destination:
                reward -= self.__wrong_destination_penalty
            else:
                reward += self.__arrival_bonus
        return reward

    @property
    def ready_to_act(self):
        """property that returns the attribute indicating if the learning agent controlling the vehicle can take its 
        next action

        Returns:
            bool: attribute indicating if the vehicle is ready to act
        """
        return self.__ready_to_act

    @property
    def current_link(self):
        """Property that returns the vehicle's current link

        Raises:
            RuntimeError: the method raises a RuntimeError if the vehicle hasn't departed yet

        Returns:
            str: vehicle's current link id
        """
        if not self.departed:
            raise RuntimeError("Vehicle hasn't departed yet!")
        return self.__current_link

    def update_current_link(self, link, current_time):
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
        destination_node = self.__environment.get_link_destination(self.__current_link)
        self.__ready_to_act = (destination_node != self.__destination and 
                                not self.__environment.is_border_node(destination_node))

    def __compute_last_link_travel_time(self, current_time):
        """Method that computes the travel time taken in last link traveled using time the vehicle departed in the link
        and the current time.

        Args:
            current_time (int): current time step
        """ 
        self.__travel_time_last_link = current_time - self.__last_link_departure_time

    def set_arrival(self, current_time):
        """Method that sets the arrival time according to the current time step given.

        Args:
            current_time (int): current time step

        Raises:
            RuntimeError: the method raises a RuntimeError if the vehicle hasn't departed yet
            RuntimeError: the method raises a RuntimeError if the time given is lower than departure time
        """
        if not self.departed:
            raise RuntimeError("Vehicle hasn't even departed yet!")
        elif current_time < self.__departure_time:
            raise RuntimeError("Invalid arrival time: value lower than departure time!")
        self.__arrival_time = current_time

    @property
    def reached_destination(self):
        """Property that indicates if the vehicle has reached a destination node.

        Returns:
            bool: boolean value indicating if vehicle has reached a destination node
        """
        return self.__arrival_time != -1

    @property
    def departed(self):
        """Property that indicates if the vehicle has been inserted in the network.

        Returns:
            bool: boolean value indicating if the vehicle has departed in the network.
        """
        return self.__departure_time != -1

    @property
    def travel_time(self):
        """Property that return the vehicle's travel time if it traveled until a destination node.

        Raises:
            RuntimeError: the method raises a RuntimeError if the vehicle hasn't departed yet
            RuntimeError: the method raises a RuntimeError if the vehicle hasn't reached a destination yet

        Returns:
            int: vehicle's route travel time
        """
        if not self.departed:
            raise RuntimeError("Vehicle hasn't departed yet!")
        elif not self.reached_destination:
            raise RuntimeError("Vehicle hasn't reached destination yet!")
        else:
            return self.__arrival_time - self.__departure_time

    def action_set(self):
        """Method that indicates to vehicle that an action has been set already.
        """
        self.__ready_to_act = False

    def __append_destination_node(self):
        """Method that appends the destination node of the vehicle's current link to its route.
        """
        destination_node = self.__environment.get_link_destination(self.__current_link)
        self.__route.append(destination_node)
