from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.dom import minidom

import numpy as np
import sumolib
import traci
import traci.constants as tc
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from script_configs.configs import NonLearnerConfig, PQLConfig, QLConfig
from sumo_ql.collector.collector import LinkCollector, ObjectiveCollector
from sumo_ql.environment.communication_device import CommunicationDevice
from sumo_ql.environment.od_pair import ODPair
from sumo_ql.environment.vehicle import Objectives, Vehicle

MAX_COMPUTABLE_OD_PAIRS = 30
MAX_VEHICLE_MARGIN = 100

CONVERSION_DICT = {
    "Step": "Step",
    "Link": "Link",
    "Travel Time": tc.VAR_CURRENT_TRAVELTIME,
    "Speed": tc.LAST_STEP_MEAN_SPEED,
    "Occupancy": tc.LAST_STEP_OCCUPANCY,
    "Running Vehicles": tc.LAST_STEP_VEHICLE_NUMBER,
    "Halting Vehicles": tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
    "CO": tc.VAR_COEMISSION,
    "CO2": tc.VAR_CO2EMISSION,
    "HC": tc.VAR_HCEMISSION,
    "PMx": tc.VAR_PMXEMISSION,
    "NOx": tc.VAR_NOXEMISSION,
    "Fuel": tc.VAR_FUELCONSUMPTION
}
HALTING_SPEED = 0.1


@dataclass(frozen=True)
class EnvConfig:
    sumocfg_file: str
    simulation_steps: int
    steps_to_populate: int
    max_vehicles: int
    right_arrival_bonus: int
    wrong_arrival_penalty: int
    communication_success_rate: float
    max_comm_dev_queue_size: int
    data_collector: LinkCollector
    use_gui: bool
    objectives: list[str]
    fit_data_collect: bool
    normalize_rewards: bool
    min_toll_speed: float
    toll_penalty: int
    graph_neighbours: dict

    @classmethod
    def from_sim_config(cls, config: NonLearnerConfig | QLConfig | PQLConfig, data_collector: LinkCollector) -> EnvConfig:
        if config.sumocfg is None:
            raise RuntimeError("Sumo cfg files should have a valid path!")
        match config:
            case NonLearnerConfig(_):
                return cls(sumocfg_file=config.sumocfg,
                           simulation_steps=config.steps,
                           steps_to_populate=config.steps,
                           max_vehicles=config.demand,
                           right_arrival_bonus=0,
                           wrong_arrival_penalty=0,
                           communication_success_rate=0.0,
                           max_comm_dev_queue_size=0,
                           data_collector=data_collector,
                           use_gui=config.gui,
                           objectives=config.observe_list,
                           fit_data_collect=False,
                           normalize_rewards=False,
                           min_toll_speed=np.infty,
                           toll_penalty=0,
                           graph_neighbours=dict())
            case QLConfig(_):
                print(f"Optimizing: {config.objective}")
                return cls(sumocfg_file=config.sumocfg,
                           simulation_steps=config.steps,
                           steps_to_populate=config.steps,
                           max_vehicles=config.demand,
                           right_arrival_bonus=config.bonus,
                           wrong_arrival_penalty=config.penalty,
                           communication_success_rate=config.communication.success_rate,
                           max_comm_dev_queue_size=config.communication.queue_size,
                           data_collector=data_collector,
                           use_gui=config.gui,
                           objectives=[config.objective],
                           fit_data_collect=config.collect_rewards,
                           normalize_rewards=config.normalize_rewards,
                           min_toll_speed=config.toll_speed,
                           toll_penalty=config.toll_value,
                           graph_neighbours=dict())
            case PQLConfig(_):
                print(f"Optimizing: {config.objectives}")
                return cls(sumocfg_file=config.sumocfg,
                           simulation_steps=config.steps,
                           steps_to_populate=config.steps,
                           max_vehicles=config.demand,
                           right_arrival_bonus=config.bonus,
                           wrong_arrival_penalty=config.penalty,
                           communication_success_rate=config.communication.success_rate,
                           max_comm_dev_queue_size=config.communication.queue_size,
                           data_collector=data_collector,
                           use_gui=config.gui,
                           objectives=config.objectives,
                           fit_data_collect=config.collect_rewards,
                           normalize_rewards=config.normalize_rewards,
                           min_toll_speed=config.toll_speed,
                           toll_penalty=config.toll_value,
                           graph_neighbours=dict())


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
            objectives (list[str], optional): Objectives the vehicles should compute so the agent can retrieve for the
            learning process. Defaults to Objective instance using only travel time.
            fit_data_collect (bool, optional): Flag that determines if the run is only for collecting reward data to use
            in future experiments (usually to normalize rewards). Defaults to False.
        """

    def __init__(self, config: EnvConfig) -> None:
        self.__sumocfg_file = config.sumocfg_file
        self.__graph_neighbours = config.graph_neighbours
        self.__network_file = self.__get_xml_filename('net-file')
        self.__route_file = self.__get_xml_filename('route-files')
        self.__network: sumolib.net.Net = sumolib.net.readNet(self.__network_file)
        self.__simulation_time = config.simulation_steps
        self.__current_step = 0
        self.__max_vehicles_running = config.max_vehicles
        self.__steps_to_populate = config.steps_to_populate
        self.__link_collector = config.data_collector
        self.__action_space: dict[str, spaces.Discrete] = dict()
        self.__comm_dev: dict[str, CommunicationDevice] = dict()
        self.__vehicles: dict[str, Vehicle] = dict()
        self.__od_pairs: dict[str, ODPair] = dict()
        self.__current_running_vehicles_n = 0
        self.__observations: dict[str, dict] = dict()
        self.__loaded_vehicles: list[str] = list()
        self.__objectives: Objectives = Objectives(config.objectives)
        self.__data_fit = None
        self.__normalize_rewards = config.normalize_rewards

        network_filepath = Path(self.__sumocfg_file[:self.__sumocfg_file.rfind('/')])
        if config.fit_data_collect:
            self.__data_fit = ObjectiveCollector(self.__objectives.objectives_str_list, network_filepath)
        if 'LIBSUMO_AS_TRACI' in os.environ and config.use_gui:
            print("Warning: using libsumo as traci can't be performed with GUI. Using sumo without GUI instead.")
            self.__sumo_bin = sumolib.checkBinary('sumo')
        else:
            self.__sumo_bin = sumolib.checkBinary('sumo-gui') if config.use_gui else sumolib.checkBinary('sumo')

        for node in self.__network.getNodes():
            self.__comm_dev[node.getID()] = CommunicationDevice(node, config.max_comm_dev_queue_size,
                                                                config.communication_success_rate, self)
            self.__action_space[node.getID()] = spaces.Discrete(len(node.getOutgoing()))

        self.__vehicles, self.__od_pairs = self.__instantiate_vehicles_and_od_pairs(config.right_arrival_bonus,
                                                                                    config.wrong_arrival_penalty,
                                                                                    config.min_toll_speed,
                                                                                    config.toll_penalty)

        print(f"Objectives: {len(self.__objectives.known_objectives)}")

    def reset(self):
        self.__link_collector.reset()
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

        if self.__using_od_pairs:
            for od_pair in self.__od_pairs.values():
                od_pair.reset()

        for vehicle_id, vehicle in self.__vehicles.items():
            vehicle.reset()
            route_id = f"r_{vehicle_id}"
            traci.route.add(route_id, vehicle.original_route)
            self.__od_pairs[vehicle.od_pair].increase_load(vehicle_id)

        subs_params = [CONVERSION_DICT[param] for param in self.__link_collector.watched_params[2:]]
        for edge in self.__network.getEdges():
            traci.edge.subscribe(edge.getID(), subs_params)

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
        self.__link_collector.save()
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
            str: Node ID that is the destination of the link.
        """
        return self.__network.getEdge(link_id).getToNode().getID()

    def get_graph_neighbours(self):
        """Method that returns dictionary of graph neighbours"""

        return self.__graph_neighbours

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
    def current_step(self) -> int:
        """Property that returns the simulation's current step.

        Returns:
            int: current step.
        """
        return self.__current_step

    @property
    def objectives(self) -> Objectives:
        """Property that returns the objectives structure use to store all agent's objectives.

        Returns:
            Objectives: structure that holds objectives used in the simulation.
        """
        return self.__objectives

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
                                            wrong_arrival_penalty: int,
                                            min_toll_speed: float,
                                            toll_penalty: int) -> tuple[dict, dict]:
        """Method that creates the vehicles classes using information from the route file. This method also creates the
        OD-pair structures that hold information about each OD-pair of the route files.

        Args:
            right_arrival_bonus (int): Bonus the vehicle receives if reached the right destination.
            wrong_arrival_penalty (int): Penalty the vehicle receives if reached the wrong destination.

        Returns:
            Union[dict, dict]: dictionaries containing, respectively, the vehicles and OD-pairs.
        """
        vehicles_dict: dict[str, Vehicle] = dict()
        od_pairs_dict: dict[str, ODPair] = dict()
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
                od_pairs_dict[od_pair] = ODPair(float(np.linalg.norm(origin_pos - destination_pos)))
                total_distance += od_pairs_dict[od_pair].straight_distance

            od_pairs_dict[od_pair].append_vehicle(vehicle_id)
            vehicles_dict[vehicle_id] = Vehicle(vehicle_id, origin_id, destination_id, right_arrival_bonus,
                                                wrong_arrival_penalty, route, self, self.__objectives,
                                                min_toll_speed, toll_penalty)

            self.__observations[vehicle_id] = {'reinserted': False,
                                               'ready_to_act': False,
                                               'origin': origin_id}

        if self.__using_od_pairs:
            for od_pair in od_pairs_dict.values():
                od_pair.min_load = math.ceil((od_pair.straight_distance / total_distance) * self.__max_vehicles_running)

        return vehicles_dict, od_pairs_dict

    def __verify_reinsertion_necessity(self, vehicle_id: str) -> None:
        """Method that verifies if number of vehicles running is enough according to the required number and reinserts
        the vehicle with the ID passed if the current running vehicles number is lower than the required value.

        Args:
            vehicle_id (str): ID of the vehicle to be reinserted in case its necessary.
        """
        if self.__using_od_pairs:
            od_pair = self.__vehicles[vehicle_id].od_pair
            self.__od_pairs[od_pair].decrease_load(vehicle_id)
            if not self.__od_pairs[od_pair].has_enough_vehicles:
                self.__reinsert_vehicle(vehicle_id)
        elif self.__current_running_vehicles_n < self.__max_vehicles_running:
            self.__reinsert_vehicle(vehicle_id)

    def __reinsert_vehicle(self, vehicle_id: str) -> None:
        """Method that properly reinserts a vehicle given its ID.

        Args:
            vehicle_id (str): ID of the vehicle to be reinserted.
        """
        if self.__populating_network and self.__using_od_pairs:
            od_pair = self.__vehicles[vehicle_id].od_pair
            vehicle_id = self.__od_pairs[od_pair].random_vehicle()

        if self.__vehicles[vehicle_id].insert():
            self.__loaded_vehicles.append(vehicle_id)

        if self.__using_od_pairs:
            self.__od_pairs[self.__vehicles[vehicle_id].od_pair].increase_load(vehicle_id)

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

    def get_link_speed(self, link_id: str) -> float:
        """Method that receives a link ID and returns it's speed limit.

        Args:
            link_id (str): link ID

        Returns:
            float: speed limit of the given link ID.
        """
        try:
            link = self.__network.getEdge(link_id)
            return link.getSpeed()
        except IndexError:
            print(f"Warning: The link iwth id {link_id} does not exist.")
        return -1

    def __compute_actions(self, actions: dict[str, int]) -> None:
        """Method that computes all of the actions performed in the current step.

        Args:
            actions (dict[str, int]): dictionary containing all actions from the agents that performed an action in the
            current step.
        """
        for vehicle_id, action in actions.items():
            self.__observations[vehicle_id]['ready_to_act'] = False
            self.__vehicles[vehicle_id].update_route(action)

    def __update_comm_dev_info(self, link_id: str, reward: np.ndarray) -> None:
        """Method that receives a reward and a link id to update the information to the destination node CommDev about
        it.

        Args:
            link_id (str): Link ID were the reward was received
            reward (np.ndarray): reward received for taking this link.
        """
        node_id = self.get_link_destination(link_id)
        if self.__comm_dev[node_id].communication_success:
            # print(f"Reward {reward} added to {node_id}")
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

    def __handle_running_vehicles(self, running_vehicles: list[str]) -> dict:
        """Method that updates all necessary information on the vehicles that are running inside the network in the
        current step.

        Args:
            running_vehicles (list[str]): list containing the IDs of all the vehicles that are running inside the
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
                should_normalize = self.__not_collecting and self.__normalize_rewards
                rewards[vehicle_id] = self.__vehicles[vehicle_id].compute_reward(normalize=should_normalize)
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

    def __handle_arrived_vehicles(self, arrived_vehicles: list[str]) -> tuple[dict, dict]:
        """Method that updates the information on all the vehicles that finished their trips in the current step.

        Args:
            arrived_vehicles (list[str]): list containing all the ID of the vehicles that finished their trips in the
            current step.

        Returns:
            Union[dict, dict]: a dictionary containing the rewards the vehicles received from finishing their trips and
            the done dictionary informing that the 'episode' from this vehicle is done.
        """
        rewards = dict()
        done = dict()
        for vehicle_id in arrived_vehicles:
            self.__vehicles[vehicle_id].update_data(self.__current_step)
            try:
                self.__vehicles[vehicle_id].set_arrival(self.__current_step)
            except RuntimeError as error:
                print(error)
                print(self.__vehicles[vehicle_id])
            done[vehicle_id] = True

            should_normalize = self.__not_collecting and self.__normalize_rewards

            reward = self.__vehicles[vehicle_id].compute_reward(use_bonus_or_penalty=False,
                                                                normalize=should_normalize)
            self.__update_comm_dev_info(self.__vehicles[vehicle_id].current_link, reward)
            reward = self.__vehicles[vehicle_id].compute_reward(use_bonus_or_penalty=False)
            self.__update_data_fit(reward)

            rewards[vehicle_id] = self.__vehicles[vehicle_id].compute_reward(normalize=should_normalize)
            self.__retrieve_observation_states(vehicle_id)
            self.__observations[vehicle_id]['ready_to_act'] = False

            self.__verify_reinsertion_necessity(vehicle_id)

        return rewards, done

    def __handle_departed_vehicles(self, departed_vehicles: list[str]) -> None:
        """Method that updates information on all the vehicles that entered the network in the current step.

        Args:
            departed_vehicles (list[str]): list containing the IDs from all the vehicles that have just entered the
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

    def __handle_step_vehicle_updates(self) -> tuple[dict, dict]:
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
        self.__update_link_data()
        return rewards, done

    def __retrieve_available_actions(self, vehicle_id: str) -> list[int]:
        """Method that returns a list of possible actions for a given vehicle ID, based on its current link.

        Args:
            vehicle_id (str): vehicle ID to retrieve possible actions.

        Returns:
            list[int]: list containing all the possible actions for the vehicle informed.
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

    def __update_link_data(self):
        step_data = {key: [] for key in self.__link_collector.watched_params}
        key_list = list(CONVERSION_DICT.keys())
        value_list = list(CONVERSION_DICT.values())
        for edge in self.__network.getEdges():
            link_id = edge.getID()
            link_data = traci.edge.getSubscriptionResults(link_id)
            step_data["Link"].append(link_id)
            for key, value in link_data.items():
                conv_key = key_list[value_list.index(key)]
                if key == tc.VAR_CURRENT_TRAVELTIME:
                    speed = link_data[tc.LAST_STEP_MEAN_SPEED]
                    if speed > HALTING_SPEED:
                        step_data[conv_key].append(edge.getLength() / speed)
                    else:
                        step_data[conv_key].append(np.NaN)
                else:
                    step_data[conv_key].append(value)
        step_data["Step"] = [self.current_step for _ in step_data["Link"]]
        self.__link_collector.append(step_data)

    @property
    def __using_od_pairs(self) -> bool:
        return len(self.__od_pairs) < MAX_COMPUTABLE_OD_PAIRS
