class Vehicle(object):

    def __init__(self, id, origin, destination, arrival_bonus, wrong_destination_penalty, environment):
        self.__id = id
        self.__origin = origin
        self.__destination = destination
        self.__arrival_bonus = arrival_bonus
        self.__wrong_destination_penalty = wrong_destination_penalty
        self.__environment = environment
        self.reset(environment.simulation_time())

    def reset(self, current_time):
        self.__current_link = None
        self.__load_time = current_time
        self.__departure_time = -1.0
        self.__arrival_time = -1.0
        self.__last_link_departure_time = -1.0
        self.__travel_time_last_link = -1.0
        self.__route = list([self.__origin])
        self.__ready_to_act = False

    @property
    def id(self):
        return self.__id

    @property
    def od_pair(self):
        return f"{self.__origin}|{self.__destination}"

    @property
    def route(self):
        return self.__route

    def compute_reward(self):
        if self.__travel_time_last_link == -1:
            raise RuntimeError("Vehicle hasn't travel a link yet!")
        reward = - self.__travel_time_last_link
        if self.reached_destination:
            if self.__route[-1] != self.__destination:
                reward -= self.__wrong_destination_penalty
            else:
                reward += self.__arrival_bonus
        return reward

    @property
    def ready_to_act(self):
        return self.__ready_to_act

    def update_current_link(self, link, current_time):
        if not self.departed:
            if current_time < self.__load_time:
                raise RuntimeError("Invalid departure time: value lower than load time!")
            self.__departure_time = current_time
        else:
            self.__append_destination_node()
            self.__travel_time_last_link = current_time - self.__last_link_departure_time
        self.__current_link = link
        self.__last_link_departure_time = current_time
        destination_node = self.__environment.get_link_destination(self.__current_link)
        self.__ready_to_act = (destination_node != self.__destination and 
                                not self.__environment.is_border_node(destination_node))

    def set_arrival(self, current_time):
        if self.__departure_time == -1:
            raise RuntimeError("Vehicle hasn't even departed yet!")
        elif current_time < self.__departure_time:
            raise RuntimeError("Invalid arrival time: value lower than departure time!")
        self.__arrival_time = current_time
        self.__append_destination_node()

    @property
    def reached_destination(self):
        return self.__arrival_time != -1

    @property
    def departed(self):
        return self.__departure_time != -1

    @property
    def travel_time(self):
        if not self.departed:
            raise RuntimeError("Vehicle hasn't departed yet!")
        elif not self.reached_destination:
            raise RuntimeError("Vehicle hasn't reached destination yet!")
        else:
            return self.__arrival_time - self.__departure_time

    def action_set(self):
        self.__ready_to_act = False

    def __append_destination_node(self):
        destination_node = self.__environment.get_link_destination(self.__current_link)
        self.__route.append(destination_node)
