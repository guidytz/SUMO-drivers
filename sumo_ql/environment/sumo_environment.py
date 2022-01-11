import sys
import os
import math
from typing import Union
from typing import Dict
from typing import List
from xml.dom import minidom
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym import spaces
import traci
import traci.constants as tc
import sumolib

from sumo_ql.environment.communication_device import CommunicationDevice
from sumo_ql.environment.vehicle import Vehicle, Objectives
from sumo_ql.environment.od_pair import ODPair
from sumo_ql.collector.collector import MainCollector, ObjectiveCollector

MAX_COMPUTABLE_OD_PAIRS = 30
MAX_VEHICLE_MARGIN = 100

class SumoEnvironment(MultiAgentEnv):
    """Class responsible for handling the environment in which the simulation takes place.

        Args:
            sumocfg_file (str): string with the path to the .sumocfg file that holds network and route information
            simulation_time (int, optional): Time to run the simulation. Defaults to 50000.
            max_vehicles (int, optional): Number of vehicles to keep running in the simulation. Defaults to 750.
            right_arrival_bonus (int, optional): Bonus vehicles receive when arriving at the right destination.
            Defaults to 1000.
            wrong_arrival_penalty (int, optional): Penalty vehicles receive when arriving at the wrong destination.
            Defaults to 1000.
            communication_success_rate (int, optional): The rate (between 0 and 1) in which the communication with the
            CommDevs succeeds. Defaults to 1.
            max_comm_dev_queue_size (int, optional): Maximum queue size to hold information on the CommDevs.
            Defaults to 30.
            steps_to_populate (int, optional): Steps to populate the network without using the learning steps.
            Defaults to 3000.
            use_gui (bool, optional): Flag that determines if the simulation should use sumo-gui. Defaults to False.
            data_collector (DataCollector, optional): Object from class responsible for collecting the experiments data. 
            Defaults to empty DataCollector.
            objectives (List[str], optional): Objectives the vehicles should compute so the agent can retrieve for the 
            learning process. Defaults to Objective instance using only travel time.
            fit_data_collect (bool, optional): Flag that determines if the run is only for collecting reward data to use
            in future experiments (usually to normalize rewards). Defaults to False.
        """
    def __init__(self, sumocfg_file: str,
                 simulation_time: int = 50000,
                 max_vehicles: int = 750,
                 right_arrival_bonus: int = 1000,
                 wrong_arrival_penalty: int = 1000,
                 communication_success_rate: float = 1,
                 max_comm_dev_queue_size: int = 30,
                 steps_to_populate: int = 3000,
                 use_gui: bool = False,
                 data_collector: MainCollector = None,
                 objectives: List[str] = None,
                 fit_data_collect: bool = False) -> None:
        self.__sumocfg_file = sumocfg_file
        self.__network_file = self.__get_xml_filename('net-file')
        self.__route_file = self.__get_xml_filename('route-files')
        self.__network = sumolib.net.readNet(self.__network_file)
        self.__simulation_time = simulation_time
        self.__current_step = None
        self.__max_vehicles_running = max_vehicles
        self.__steps_to_populate = steps_to_populate if steps_to_populate < simulation_time else simulation_time
        self.__collector = data_collector or MainCollector() # in case of being None
        self.__action_space: Dict[spaces.Discrete] = dict()
        self.__comm_dev: Dict[str, CommunicationDevice] = dict()
        self.__vehicles: Dict[str, Vehicle] = dict()
        self.__od_pairs: Dict[str, ODPair] = dict()
        self.__current_running_vehicles_n = 0
        self.__observations: Dict[str, dict] = dict()
        self.__loaded_vehicles: List[str] = list()
        self.__objectives: Objectives = Objectives(objectives or [tc.VAR_ROAD_ID])
        self.__data_fit = None
        if fit_data_collect:
            bar_pos = self.__sumocfg_file.rfind('/')
            self.__data_fit = ObjectiveCollector(self.__objectives.objective_str, self.__sumocfg_file[:bar_pos])
        if 'LIBSUMO_AS_TRACI' in os.environ and use_gui:
            print("Warning: using libsumo as traci can't be performed with GUI. Using sumo without GUI instead.")
            self.__sumo_bin = sumolib.checkBinary('sumo')
        else:
            self.__sumo_bin = sumolib.checkBinary('sumo-gui') if use_gui else sumolib.checkBinary('sumo')

        for node in self.__network.getNodes():
            self.__comm_dev[node.getID()] = CommunicationDevice(node, max_comm_dev_queue_size,
                                                                communication_success_rate, self)
            self.__action_space[node.getID()] = spaces.Discrete(len(node.getOutgoing()))

        self.__vehicles, self.__od_pairs = self.__instantiate_vehicles_and_od_pairs(right_arrival_bonus,
                                                                                    wrong_arrival_penalty)


    def reset(self):
        self.__collector.reset()
        self.__current_step = 0
        sumo_cmd = [
            self.__sumo_bin,
            "-c", self.__sumocfg_file,
            "--max-num-vehicles", f"{self.__max_vehicles_running + MAX_VEHICLE_MARGIN}",
            "--verbose",
            "--random"
        ]
        traci.start(sumo_cmd)
        traci.simulation.subscribe((tc.VAR_ARRIVED_VEHICLES_IDS,
                                    tc.VAR_DEPARTED_VEHICLES_IDS))
        traci.vehicle.subscribe('', [tc.TRACI_ID_LIST, tc.ID_COUNT])
        for vehicle_id in self.__vehicles:
            self.__vehicles[vehicle_id].reset()
            route_id = f"r_{vehicle_id}"
            traci.route.add(route_id, self.__vehicles[vehicle_id].original_route)

        if len(self.__od_pairs) < MAX_COMPUTABLE_OD_PAIRS:
            for od_pair in self.__od_pairs:
                self.__od_pairs[od_pair].reset()

        self.__populate_network()
        return self.__observations

    def step(self, action_dict):
        rewards = dict()
        done = dict()
        self.__handle_loaded_vehicles()

        if action_dict is not None:
            self.__compute_actions(action_dict)

        self.__sumo_step()
        rewards, done = self.__handle_step_vehicle_updates()
        done.update({'__all__': self.__current_step >= self.__simulation_time})
        return self.__observations, rewards, done, {}

    def close(self) -> None:
        """Method that closes the traci run and saves collected data to csv files.
        """
        self.__collector.save()
        if self.__data_fit is not None:
            self.__data_fit.save()
        traci.close()

    def get_comm_dev(self, node_id: str) -> CommunicationDevice:
        """Method that returns a CommDev given the node it is present.

        Args:
            node_id (str): Node ID in which the CommDev is present.

        Returns:
            CommunicationDevice: CommDev present in the given node.
        """
        return self.__comm_dev[node_id]

    def get_link_origin(self, link_id: str) -> str:
        """Method that returns the origin node given the link ID.

        Args:
            link_id (str): Link ID to retrieve the origin node

        Returns:
            str: Node ID that is the origin of the link.
        """
        return self.__network.getEdge(link_id).getFromNode().getID()

    def get_link_destination(self, link_id: str) -> str:
        """Method that returns the destination node given the link ID.

        Args:
            link_id (str): Link ID to retrieve the destination node

        Returns:
            str: Node ID that is the destination   of the link.
        """
        return self.__network.getEdge(link_id).getToNode().getID()

    def is_border_node(self, node_id: str) -> bool:
        """Method that tests whether a given node is in the border of the network.

        Args:
            node_id (str): Node ID to test if it is in the border of the network

        Returns:
            bool: Boolean that returns the value of the test
        """
        for link in self.__network.getNode(node_id).getIncoming():
            if len(link.getOutgoing()) > 0:
                return False
        return True

    @property
    def action_space(self) -> dict:
        """Property that returns the action space of the environment.

        Returns:
            dict: action space computed to the network provided.
        """
        return self.__action_space

    @property
    def sim_path(self) -> str:
        """Method that retuns the simulation file path of the current run.

        Returns:
            str: string containing the path to the file of the current run simulation.
        """
        return self.__sumocfg_file[:self.__sumocfg_file.rfind('/')]

    def get_action(self, previous_state: str, next_state: str) -> int:
        """Method that returns the action (link index) given an origin and destination node.
        The method returns -1 if the action was not found.

        Args:
            previous_state (str): Node ID of the origin node
            next_state (str): Node ID of the the destination node

        Returns:
            int: action (link index) between the origin and destination nodes.
        """
        for action, link in enumerate(self.__network.getNode(previous_state).getOutgoing()):
            if self.get_link_destination(link.getID()) == next_state:
                return action
        return -1

    @property
    def __populating_network(self):
        """Property that returns a boolean that indicates if the network is beeing populated.

        Returns:
            bool: boolean that returns True if the network is beeing populated or False otherwise.
        """
        return self.__current_step < self.__steps_to_populate

    def __populate_network(self) -> None:
        """Method that performs the steps defined to populate the network with vehicles (following their original route
        defined in the route file) before considering the learning process.
        """
        while self.__populating_network:
            self.__handle_loaded_vehicles()
            self.__sumo_step()
            self.__handle_step_vehicle_updates()

    def __sumo_step(self) -> None:
        """Method that performs a simulation step.
        """
        self.__current_step += 1
        traci.simulation.step()

    def __get_xml_filename(self, attribute: str) -> str:
        """Method that retrieves the file name (for network or route) present in the .sumocfg file given the attribute.

        Args:
            attribute (str): attribute to retrieve the filename

        Returns:
            str: filename
        """
        name = self.__sumocfg_file[:self.__sumocfg_file.rfind("/")+1]
        name += minidom.parse(self.__sumocfg_file).getElementsByTagName(attribute)[0].attributes['value'].value
        return name

    def is_link(self, edge_id: str) -> bool:
        """Method that tests if the given link id is from a proper network edge;

        Args:
            edge_id (str): id to test if it is a proper network edge.

        Returns:
            bool: The result of the test.
        """
        try:
            _ = self.__network.getEdge(edge_id)
            return True
        except KeyError:
            return False

    def __instantiate_vehicles_and_od_pairs(self, right_arrival_bonus: int,
                                            wrong_arrival_penalty: int) -> Union[dict, dict]:
        """Method that creates the vehicles classes using information from the route file. This method also creates the
        OD-pair structures that hold information about each OD-pair of the route files.

        Args:
            right_arrival_bonus (int): Bonus the vehicle receives if reached the right destination.
            wrong_arrival_penalty (int): Penalty the vehicle receives if reached the wrong destination.

        Returns:
            Union[dict, dict]: dictionaries containing, respectively, the vehicles and OD-pairs.
        """
        vehicles_dict = dict()
        od_pairs_dict = dict()
        total_distance = 0

        vehicles_parse = minidom.parse(self.__route_file).getElementsByTagName('vehicle')
        for vehicle_attr in vehicles_parse:
            vehicle_id = vehicle_attr.getAttribute('id')
            route = vehicle_attr.getElementsByTagName('route')[0].getAttribute('edges').split(' ')

            origin_id = self.get_link_origin(route[0])
            destination_id = self.get_link_destination(route[-1])

            od_pair = f"{origin_id}|{destination_id}"
            if od_pair not in od_pairs_dict:
                origin_pos = np.array(self.__network.getNode(origin_id).getCoord())
                destination_pos = np.array(self.__network.getNode(destination_id).getCoord())
                od_pairs_dict[od_pair] = ODPair(np.linalg.norm(origin_pos - destination_pos))
                total_distance += od_pairs_dict[od_pair].straight_distance

            vehicles_dict[vehicle_id] = Vehicle(vehicle_id, origin_id, destination_id, right_arrival_bonus,
                                                wrong_arrival_penalty, route, self, self.__objectives)

            self.__observations[vehicle_id] = {'reinserted': False,
                                               'ready_to_act': False,
                                               'origin': origin_id}

        if len(self.__od_pairs) < MAX_COMPUTABLE_OD_PAIRS:
            for od_pair in od_pairs_dict:
                od_pairs_dict[od_pair].min_load = math.ceil((od_pairs_dict[od_pair].straight_distance / total_distance)
                                                            * self.__max_vehicles_running)

        return vehicles_dict, od_pairs_dict

    def __verify_reinsertion_necessity(self, vehicle_id: str) -> None:
        """Method that verifies if number of vehicles running is enough according to the required number and reinserts
        the vehicle with the ID passed if the current running vehicles number is lower than the required value.

        Args:
            vehicle_id (str): ID of the vehicle to be reinserted in case its necessary.
        """
        if len(self.__od_pairs) < MAX_COMPUTABLE_OD_PAIRS:
            od_pair = self.__vehicles[vehicle_id].od_pair
            self.__od_pairs[od_pair].decrease_load()
            if not self.__od_pairs[od_pair].has_enough_vehicles:
                self.__reinsert_vehicle(vehicle_id)
        elif self.__current_running_vehicles_n < self.__max_vehicles_running:
            self.__reinsert_vehicle(vehicle_id)

    def __reinsert_vehicle(self, vehicle_id: str) -> None:
        """Method that properly reinserts a vehicle given its ID.

        Args:
            vehicle_id (str): ID of the vehicle to be reinserted.
        """
        if self.__vehicles[vehicle_id].insert():
            self.__loaded_vehicles.append(vehicle_id)

        if len(self.__od_pairs) < MAX_COMPUTABLE_OD_PAIRS:
            self.__od_pairs[self.__vehicles[vehicle_id].od_pair].increase_load()

    def get_action_link(self, node_id: str, action: int) -> str:
        """Method that returns the link that corresponds to the action performed, given a node/state where it was
        performed.

        Args:
            node_id (str): Node ID where the action was performed.
            action (int): The action performed.

        Returns:
            str: Link ID that corresponds to the action performed.
        """
        node = self.__network.getNode(node_id)
        outgoing = node.getOutgoing()
        link_ids = [link.getID() for link in outgoing]
        try:
            _ = link_ids[action]
        except IndexError:
            print("Aborting simulation: action could not be chosen due to index error")
            print(f"Possible links: {link_ids}")
            print(f"Action -{action}- tried should correspond to an index in the list above.")
            print(f"Node ID: {node.getID()}")
            self.close()
            sys.exit()
        return link_ids[action]

    def __compute_actions(self, actions: Dict[str, int]) -> None:
        """Method that computes all of the actions performed in the current step.

        Args:
            actions (Dict[str, int]): dictionary containing all actions from the agents that performed an action in the
            current step.
        """
        for vehicle_id, action in actions.items():
            self.__observations[vehicle_id]['ready_to_act'] = False
            self.__vehicles[vehicle_id].update_route(action)

    def __update_comm_dev_info(self, link_id: str, reward: int) -> None:
        """Method that receives a reward and a link id to update the information to the destination node CommDev about
        it.

        Args:
            link_id (str): Link ID were the reward was received
            reward (int): reward received for taking this link.
        """
        node_id = self.get_link_destination(link_id)
        if self.__comm_dev[node_id].communication_success:
            self.__comm_dev[node_id].update_stored_rewards(link_id, reward)

    def __handle_loaded_vehicles(self) -> None:
        """Method that updates information necessary in the observation for all the vehicles loaded in the simulation
        in the current step.
        """
        for vehicle_id in self.__loaded_vehicles:
            self.__vehicles[vehicle_id].reset()
            self.__observations[vehicle_id]['reinserted'] = True
            self.__observations[vehicle_id]['ready_to_act'] = False
            self.__observations[vehicle_id]['current_state'] = self.__vehicles[vehicle_id].origin
            self.__observations[vehicle_id]['previous_state'] = None
        self.__loaded_vehicles.clear()

    def __handle_running_vehicles(self, running_vehicles: List[str]) -> dict:
        """Method that updates all necessary information on the vehicles that are running inside the network in the
        current step.

        Args:
            running_vehicles (List[str]): list containing the IDs of all the vehicles that are running inside the
            network in the current step.

        Returns:
            dict: dictionary with the rewards from all the vehicles that finished running through a link in the current
            step.
        """
        rewards = dict()
        for vehicle_id in running_vehicles:
            self.__vehicles[vehicle_id].update_data(self.__current_step)
            if self.__vehicles[vehicle_id].changed_link:
                vehicle_last_link = self.__vehicles[vehicle_id].last_link
                rewards[vehicle_id] = self.__vehicles[vehicle_id].compute_reward(normalize=self.__not_collecting)
                self.__update_data_fit(rewards[vehicle_id])

                self.__update_comm_dev_info(vehicle_last_link, rewards[vehicle_id])

                self.__retrieve_observation_states(vehicle_id)
                self.__observations[vehicle_id]['last_link_state'] = self.get_link_origin(vehicle_last_link)
                if self.__vehicles[vehicle_id].ready_to_act:
                    self.__observations[vehicle_id]['ready_to_act'] = True
                    self.__observations[vehicle_id]['available_actions'] = self.__retrieve_available_actions(vehicle_id)
                else:
                    self.__observations[vehicle_id]['ready_to_act'] = False
                    self.__observations[vehicle_id]['available_actions'] = []

        return rewards

    def __handle_arrived_vehicles(self, arrived_vehicles: List[str]) -> Union[dict, dict]:
        """Method that updates the information on all the vehicles that finished their trips in the current step.

        Args:
            arrived_vehicles (List[str]): list containing all the ID of the vehicles that finished their trips in the
            current step.

        Returns:
            Union[dict, dict]: a dictionary containing the rewards the vehicles received from finishing their trips and
            the done dictionary informing that the 'episode' from this vehicle is done.
        """
        rewards = dict()
        done = dict()
        data_collected = list()
        for vehicle_id in arrived_vehicles:
            self.__vehicles[vehicle_id].update_data(self.__current_step)
            try:
                self.__vehicles[vehicle_id].set_arrival(self.__current_step)
            except RuntimeError as error:
                print(error)
                print(self.__vehicles[vehicle_id])
            done[vehicle_id] = True

            reward = self.__vehicles[vehicle_id].compute_reward(use_bonus_or_penalty=False,
                                                                normalize=self.__not_collecting)
            self.__update_comm_dev_info(self.__vehicles[vehicle_id].current_link, reward)
            reward = self.__vehicles[vehicle_id].compute_reward(use_bonus_or_penalty=False)
            self.__update_data_fit(reward)

            rewards[vehicle_id] = self.__vehicles[vehicle_id].compute_reward(normalize=self.__not_collecting)
            self.__retrieve_observation_states(vehicle_id)
            self.__observations[vehicle_id]['ready_to_act'] = False
            if self.__vehicles[vehicle_id].is_correct_arrival:
                data_collected.append(self.__vehicles[vehicle_id].cumulative_data)

            self.__verify_reinsertion_necessity(vehicle_id)

        if len(data_collected) > 1 or self.__collector.time_to_measure(self.__current_step):
            self.__collector.append_list(data_collected, self.__current_step)

        return rewards, done

    def __handle_departed_vehicles(self, departed_vehicles: List[str]) -> None:
        """Method that updates information on all the vehicles that entered the network in the current step.

        Args:
            departed_vehicles (List[str]): list containing the IDs from all the vehicles that have just entered the
            network in the current step.
        """
        for vehicle_id in departed_vehicles:
            self.__vehicles[vehicle_id].departure(self.__current_step)
            if self.__vehicles[vehicle_id].ready_to_act:
                self.__observations[vehicle_id]['reinserted'] = False
                self.__observations[vehicle_id]['ready_to_act'] = True
                self.__retrieve_observation_states(vehicle_id)
                self.__observations[vehicle_id]['available_actions'] = self.__retrieve_available_actions(vehicle_id)
            else:
                self.__observations[vehicle_id]['ready_to_act'] = False
                self.__observations[vehicle_id]['available_actions'] = []

    def __handle_step_vehicle_updates(self) -> Union[dict, dict]:
        """Method that collects information on the vehicles from traci in the current step and performs the methods
        that update this data correctly.

        Returns:
            Union[dict, dict]: a dictionary containing the rewards from all the vehicles that received a reward in the
            current step and a dictionary containing a flag that indicates if the 'episode' is done for each vehicle.
        """
        traci_simulation_info = traci.simulation.getSubscriptionResults()
        traci_vehicles_info = traci.vehicle.getSubscriptionResults('')
        self.__current_running_vehicles_n = traci_vehicles_info[tc.ID_COUNT]
        running_vehicles = traci_vehicles_info[tc.TRACI_ID_LIST]
        arrived_vehicles = traci_simulation_info[tc.VAR_ARRIVED_VEHICLES_IDS]
        departed_vehicles = traci_simulation_info[tc.VAR_DEPARTED_VEHICLES_IDS]
        rewards = dict()

        self.__handle_departed_vehicles(departed_vehicles)
        rewards = self.__handle_running_vehicles(running_vehicles)
        arrived_rewards, done = self.__handle_arrived_vehicles(arrived_vehicles)
        rewards.update(arrived_rewards)
        return rewards, done

    def __retrieve_available_actions(self, vehicle_id: str) -> List[int]:
        """Method that returns a list of possible actions for a given vehicle ID, based on its current link.

        Args:
            vehicle_id (str): vehicle ID to retrieve possible actions.

        Returns:
            List[int]: list containing all the possible actions for the vehicle informed.
        """
        available_links = self.__network.getEdge(self.__vehicles[vehicle_id].current_link).getOutgoing()
        available_actions = list()
        action_link = enumerate(self.__network.getNode(self.__vehicles[vehicle_id].route[-1]).getOutgoing())
        for action, link in action_link:
            if link in available_links:
                available_actions.append(action)

        return available_actions

    def __retrieve_observation_states(self, vehicle_id: str) -> None:
        """Method that updates the current and previous states of a given vehicle in the observation dictionary.

        Args:
            vehicle_id (str): vehicle ID to update the info.
        """
        current_link = self.__vehicles[vehicle_id].current_link
        self.__observations[vehicle_id]['current_state'] = self.get_link_destination(current_link)
        self.__observations[vehicle_id]['previous_state'] = self.get_link_origin(current_link)

    def __update_data_fit(self, reward):
        if self.__data_fit is not None:
            self.__data_fit.append_rewards([reward])

    @property
    def __not_collecting(self):
        return self.__data_fit is None
