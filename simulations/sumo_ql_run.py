import sys
import argparse
from typing import Dict

from sumo_ql.environment.sumo_environment import SumoEnvironment
from sumo_ql.agent.q_learning import QLAgent
from sumo_ql.exploration.epsilon_greedy import EpsilonGreedy
from sumo_ql.collector.collector import DataCollector


class SumoQLRun:
    """Class responsible for running the simulations.

    Args:
        sumocfg_file (str): string with the path to the .sumocfg file that holds network and route information
        simulation_time (int): Time to run the simulation.
        max_vehicles (int): Number of vehicles to keep running in the simulation.
        right_arrival_bonus (int): Bonus vehicles receive when arriving at the right destination.
        wrong_arrival_penalty (int): Penalty vehicles receive when arriving at the wrong destination.
        communication_success_rate (float): The rate (between 0 and 1) in which the communication with the CommDevs
        succeeds.
        max_comm_dev_queue_size (int): Maximum queue size to hold information on the CommDevs.
        steps_to_populate (int): Steps to populate the network without using the learning steps.
        moving_average_gap (int): Step gap to take the travel times moving average measurement.
        use_gui (bool): Flag that determines if the simulation should use sumo-gui.
    """

    def __init__(self, sumocfg_file: str,
                 simulation_time: int,
                 max_vehicles: int,
                 right_arrival_bonus: int,
                 wrong_arrival_penalty: int,
                 communication_success_rate: float,
                 max_comm_dev_queue_size: int,
                 steps_to_populate: int,
                 moving_average_gap: int,
                 use_gui: bool) -> None:
        data_collector = self.__generate_data_collector(sumocfg_file,
                                                        simulation_time,
                                                        steps_to_populate,
                                                        communication_success_rate,
                                                        moving_average_gap)
        self.__agents = None
        self.__observations = None
        self.__rewards = None

        self.__env = SumoEnvironment(sumocfg_file,
                                     simulation_time,
                                     max_vehicles,
                                     right_arrival_bonus,
                                     wrong_arrival_penalty,
                                     communication_success_rate,
                                     max_comm_dev_queue_size,
                                     steps_to_populate,
                                     use_gui,
                                     data_collector=data_collector)

    def __generate_data_collector(self, cfgfile: str,
                                  sim_steps: int,
                                  pop_steps: int,
                                  comm_succ_rate: float,
                                  moving_avg_gap: int) -> DataCollector:
        """Method that generates a data collector based on the information used in the simulation.

        Args:
            cfgfile (str): string with the path to the .sumocfg file that holds network and route information
            sim_steps (int): Time to run the simulation.
            comm_succ_rate (float): The rate (between 0 and 1) in which the communication with the CommDevs succeeds.
            pop_steps (int): Steps to populate the network without using the learning steps.
            moving_avg_gap (int): Step gap to take the travel times moving average measurement.

        Returns:
            DataCollector: class responsible for collecting data from the environment.
        """
        main_simulation_name = str(cfgfile).split('/')[-2]
        additional_folders = list()

        learning_folder = "learning" if pop_steps < sim_steps else "not_learning"
        additional_folders.append(learning_folder)

        if learning_folder == "learning":
            c2i_sr_folder = f"C2I_sr{int(comm_succ_rate * 100)}"
            additional_folders.append(c2i_sr_folder)

        steps_folder = f"steps_{sim_steps // 1000}K"
        additional_folders.append(steps_folder)

        return DataCollector(sim_filename=main_simulation_name,
                             steps_to_measure=moving_avg_gap,
                             additional_folders=additional_folders)

    def run(self) -> None:
        """Method that runs a simulation.
        """
        self.__observations = self.__env.reset()
        self.__agents: Dict[str, QLAgent] = dict()
        done = {'__all__': False}
        while not done['__all__']:
            actions = dict()
            for vehicle_id in self.__observations:
                if self.__observations[vehicle_id]['reinserted'] and vehicle_id not in self.__agents:
                    self.__create_agent(vehicle_id)

            for vehicle_id in self.__observations:
                if self.__observations[vehicle_id]['ready_to_act'] and vehicle_id in self.__agents:
                    self.__handle_communication(vehicle_id, self.__observations[vehicle_id]['current_state'])
                    current_state = self.__observations[vehicle_id]['current_state']
                    available_actions = self.__observations[vehicle_id]['available_actions']
                    actions[vehicle_id] = self.__agents[vehicle_id].act(current_state, available_actions)

            self.__observations, self.__rewards, done, _ = self.__env.step(actions)

            for vehicle_id, reward in self.__rewards.items():
                if vehicle_id in self.__agents:
                    if vehicle_id in done:
                        previous_state = self.__observations[vehicle_id]['previous_state']
                        next_state = self.__observations[vehicle_id]['current_state']
                        self.__handle_learning(vehicle_id, previous_state, next_state, reward)
                    else:
                        previous_state = self.__observations[vehicle_id]['last_link_state']
                        next_state = self.__observations[vehicle_id]['previous_state']
                        self.__handle_learning(vehicle_id, previous_state, next_state, reward)
        self.__env.close()

    def __create_agent(self, vehicle_id: str) -> None:
        """Method that creates a learning agent and puts it in the agents dictionary.

        Args:
            vehicle_id (str): vehicle id to identify the agent.
        """
        self.__agents[vehicle_id] = QLAgent(action_space=self.__env.action_space,
                                            exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.05))

    def __handle_communication(self, vehicle_id: str, state: str):
        """Method that retrieves CommDevs information if the C2I communication succeeds to update the agent's knowledge
        about the network.

        Args:
            vehicle_id (str): ID of the vehicle that will communicate with the CommDev.
            state (str): the state the CommDev is present.
        """
        comm_dev = self.__env.get_comm_dev(state)
        if comm_dev.communication_success:
            expected_rewards = comm_dev.get_outgoing_links_expected_rewards()
            for link, expected_reward in expected_rewards.items():
                origin = self.__env.get_link_origin(link)
                destination = self.__env.get_link_destination(link)
                self.__handle_learning(vehicle_id, origin, destination, expected_reward)

    def __handle_learning(self, vehicle_id: str, origin_node: str, destination_node: str, reward: int):
        """Method that takes care of the learning process for the agent given.

        Args:
            vehicle_id (str): ID of the vehicle to process learning.
            origin_node (str): origin node the agent took the action.
            destination_node (str): destination node the action leaded to.
            reward (int): reward received from the action taken.

        Raises:
            Exception: it raises an Exception if anything goes wrong.
        """
        try:
            action = self.__env.get_action(origin_node, destination_node)
            self.__agents[vehicle_id].learn(action, origin_node, destination_node, reward)
        except Exception as exception:
            print(f"{vehicle_id = }")
            print(f"{self.__observations = }")
            print(f"{self.__rewards = }")
            raise Exception(exception).with_traceback(exception.__traceback__)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(prog='Script to run SUMO environment with multiagent Q-Learning algorithm')

    parse.add_argument("-c", "--cfg-file", action="store", dest="cfgfile",
                       help="define the config SUMO file (mandatory)")
    parse.add_argument("-d", "--demand", action="store", type=int, dest="demand",
                       default=750, help="desired network demand (default = 750)")
    parse.add_argument("-s", "--steps", action="store", type=int, default=60000,
                       help="number of max steps (default = 60000)", dest="steps")
    parse.add_argument("-w", "--wait-learning", action="store", type=int, default=3000, dest="wait_learn",
                       help="Time steps before agents start the learning (default = 3000)")
    parse.add_argument("-g", "--gui", action="store_true", dest="gui", default=False,
                       help="uses SUMO GUI instead of CLI")
    parse.add_argument("-m", "--mav", action="store", type=int, dest="mav", default=100,
                       help="Moving gap size (default = 100 steps)")
    parse.add_argument("-r", "--success-rate", action="store", type=float, dest="comm_succ_rate", default=1,
                       help="Communication success rate (default = 1)")
    parse.add_argument("-q", "--queue-size", action="store", type=int, dest="queue_size", default=30,
                       help="CommDev queue size (default = 30)")
    parse.add_argument("-b", "--bonus", action="store", type=int, dest="bonus", default=1000,
                       help="Bonus agents receive by finishing their trip at the right destination (default = 1000)")
    parse.add_argument("-p", "--penalty", action="store", type=int, dest="penalty", default=1000,
                       help="Penalty agents receive by finishing their trip at the wrong destination (default = 1000)")

    options = parse.parse_args()
    if not options.cfgfile:
        print('Wrong usage of script!')
        print()
        parse.print_help()
        sys.exit()

    application = SumoQLRun(sumocfg_file=options.cfgfile,
                            simulation_time=options.steps,
                            max_vehicles=options.demand,
                            right_arrival_bonus=options.bonus,
                            wrong_arrival_penalty=options.penalty,
                            communication_success_rate=options.comm_succ_rate,
                            max_comm_dev_queue_size=options.queue_size,
                            steps_to_populate=options.wait_learn,
                            moving_average_gap=options.mav,
                            use_gui=options.gui)
    application.run()
