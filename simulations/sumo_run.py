import logging
import os
import pickle
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from script_configs import create_parser
from script_configs.configs import NonLearnerConfig, PQLConfig, QLConfig
from sumo_ql.agent.q_learning import PQLAgent, QLAgent
from sumo_ql.collector.collector import DefaultCollector, LinkCollector
from sumo_ql.environment.sumo_environment import EnvConfig, SumoEnvironment
from sumo_ql.exploration.epsilon_greedy import EpsilonGreedy
from sumo_vg.virtual_graph import generate_graph_neighbours_dict

SAVE_OBJ_CHOSEN = False


def run_sim(config: NonLearnerConfig | QLConfig | PQLConfig, date: datetime = datetime.now(), iteration: int = -1):
    """Function used to run the simulations, given a set of arguments passed to the script and the iteration (run
    number).

    Args:
        args (argparse.Namespace): namespace containing all the arguments passed to the script
        date (datetime): datetime object that indicates the simulations beggining. Defaults to datetime.now().
        iteration (int, optional): Iteration of simulation run (necessary for log purposes on multiple runs).
        Defaults to -1 (when running only one simulation, then the iteration number is discarded).

    Raises:
        OSError: the function raises an OSError if the log directory can't be created.
        Exception: If any unknown error occurs during the simulation, it raises an exception.
    """
    agents: dict[str, QLAgent | PQLAgent] = dict()
    observations = None
    rewards = None
    env: SumoEnvironment

    if isinstance(config, NonLearnerConfig) or config.virtual_graph is None:
        uses_virtual_graph = False
        print("Not using virtual graph")
        graph_neighbours_dict = {}
    else:
        print("Using virtual graph")
        uses_virtual_graph = True
        if config.virtual_graph.file is None:
            # reads pickle file containing virtual graph neighbours dictionary
            print("Reading graph neighbours dictionary from pickle file...")
            with open(f"{config.virtual_graph.vg_dict}", "rb") as vg_dict_pickle:
                graph_neighbours_dict = pickle.load(vg_dict_pickle, encoding="bytes")
        else:
            # generates graph neighbours dict
            print("Generating graph neighbours dictionary...")
            network_name = str(config.sumocfg).split('/')[-2]
            vg_neighbours_dict = generate_graph_neighbours_dict(config.virtual_graph.file,
                                                                   config.virtual_graph.attributes,
                                                                   config.virtual_graph.labels,
                                                                   config.virtual_graph.restrictions,
                                                                   config.virtual_graph.threshold,
                                                                   config.virtual_graph.use_or,
                                                                   config.virtual_graph.measures,
                                                                   config.virtual_graph.no_image,
                                                                   config.virtual_graph.raw,
                                                                   config.virtual_graph.giant,
                                                                   config.virtual_graph.not_normalize,
                                                                   config.virtual_graph.min_degree,
                                                                   config.virtual_graph.min_step,
                                                                   arestas_para_custoso=2000,
                                                                   precisao=10,
                                                                   intervalo_vizinhos=config.virtual_graph.interval,
                                                                   network_name=network_name,
                                                                   vertex_attribute=config.virtual_graph.vertex_attribute)

    def create_log(dirname: str, date: datetime) -> None:
        """Method that creates a log file that has information of beginning and end of simulations when making multiple
        runs.

        Args:
            dirname (str): directory name where the log will be saved (within the log directory).
            date (datetime): datetime object that is used to know when the multiple runs started.

        Raises:
            OSError: the method raises an OSError if the directory couldn't be created (it doesn't raise the error if
            the directory already exists).
        """

        log_directory = Path(f"log/{dirname}")
        log_directory.mkdir(exist_ok=True, parents=True)

        logging.basicConfig(format='%(asctime)s: %(message)s',
                            datefmt='%d-%m-%Y %H:%M:%S',
                            filename=f'log/{dirname}/mult_sims_{date.strftime("%d-%m-%y_%H-%M-%S")}.log',
                            level=logging.INFO)

    def run_nonlearner(config: NonLearnerConfig, date: datetime, iteration: int):
        pass

    def run_ql(config: QLConfig, date: datetime, iteration: int):
        pass

    def run_pql(config: PQLConfig, date: datetime, iteration: int):
        pass

    def generate_data_collector(cfgfile: str,
                                sim_steps: int,
                                pop_steps: int,
                                comm_succ_rate: float,
                                moving_avg_gap: int,
                                date: datetime,
                                uses_virtual_graph: bool,
                                agent_type: str,
                                n_runs: int = 1,
                                observe_list: list[str] | None = None) -> LinkCollector:
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
        print(f"{observe_list = }")
        if observe_list is None:
            raise RuntimeError("Objectives list cannot be empty!")

        main_simulation_name = str(cfgfile).split('/')[-2]
        additional_folders = list()

        learning_folder = Path("not_learning")
        if pop_steps < sim_steps:
            learning_folder = Path(agent_type)
        additional_folders.append(learning_folder)

        if learning_folder != Path("not_learning"):
            c2i_sr_folder = Path(f"C2I_sr{int(comm_succ_rate * 100)}")
            additional_folders.append(c2i_sr_folder)

        steps_folder = Path(f"steps_{sim_steps // 1000}K")
        additional_folders.append(steps_folder)

        objectives_folder = Path(f"opt_{'_'.join(observe_list)}")
        additional_folders.append(objectives_folder)

        vg_folder = Path("virtual_graph") if uses_virtual_graph else Path("no_virtual_graph")
        additional_folders.append(vg_folder)

        if n_runs > 1:
            batch_folder = Path(f"batch_{date.strftime('%H-%M')}_{n_runs}_runs")
            additional_folders.append(batch_folder)
            create_log(main_simulation_name, date)

        return LinkCollector(network_name=main_simulation_name,
                             aggregation_interval=moving_avg_gap,
                             additional_folders=additional_folders,
                             params=observe_list,
                             date=date)

    def create_environment(config: NonLearnerConfig | QLConfig | PQLConfig, graph_neighbours_dict: dict) -> SumoEnvironment:
        """Method that creates a SUMO environment given the arguments necessary to it.

        Args:
            args (argparse.Namespace): namespace that contains the arguments passed to the script.

        Returns:
            SumoEnvironment: an environment object used in the learning process.
        """
        match config:
            case NonLearnerConfig(_):
                pop_steps = config.steps
                comm_success_rate = 0
                agent_type = "not_learning"
            case QLConfig(_):
                pop_steps = config.wait_learn
                comm_success_rate = config.communication.success_rate
                agent_type = "QL"
            case PQLConfig(_):
                pop_steps = config.wait_learn
                comm_success_rate = config.communication.success_rate
                agent_type = "PQL"

        if config.sumocfg is None:
            raise ValueError("Sumo cfg file should not be none here")

        data_collector = generate_data_collector(cfgfile=config.sumocfg,
                                                 sim_steps=config.steps,
                                                 pop_steps=pop_steps,
                                                 comm_succ_rate=comm_success_rate,
                                                 moving_avg_gap=config.aw,
                                                 date=date,
                                                 uses_virtual_graph=uses_virtual_graph,
                                                 agent_type=agent_type,
                                                 n_runs=config.nruns,
                                                 observe_list=config.observe_list)

        env_config = EnvConfig.from_sim_config(config, data_collector)
        environment = SumoEnvironment(env_config)
        return environment

    def run(iteration) -> None:
        """Method that runs a simulation.
        """
        chosen_obj_collector: DefaultCollector | None = None
        if iteration != -1:
            logging.info("Iteration %s started.", iteration)
        observations = env.reset()
        done = {'__all__': False}

        if isinstance(config, PQLConfig):
            network_name = str(config.sumocfg).split('/')[-2]
            chosen_obj_collector = DefaultCollector(1,
                                                    Path("results/ChosenObj/") /
                                                    f"{network_name}" /
                                                    f"{date.strftime('%y_%m_%d')}",
                                                    ["Step"] + config.objectives)

        while not done['__all__']:
            actions: dict[str, int] = dict()
            for vehicle_id, vehicle in observations.items():
                if vehicle['reinserted'] and vehicle_id not in agents:
                    create_agent(vehicle_id)

            chosen_sum = [0 for _ in range(len(config.objectives))] if isinstance(config, PQLConfig) else []
            for vehicle_id, vehicle in observations.items():
                if vehicle['ready_to_act'] and vehicle_id in agents:
                    handle_communication(vehicle_id, vehicle['current_state'])
                    current_state = vehicle['current_state']
                    available_actions = vehicle['available_actions']
                    match agents:
                        case dict(QLAgent(_)):
                            actions[vehicle_id] = agents[vehicle_id].act(current_state, available_actions)
                        case dict(PQLAgent(_)):
                            actions[vehicle_id], chosen_obj = agents[vehicle_id].act(current_state,
                                                                                     available_actions)
                            if chosen_obj != -1:
                                chosen_sum[chosen_obj] += 1

            if isinstance(config, PQLConfig):
                match chosen_obj_collector:
                    case None:
                        raise RuntimeError("Collector for chosen objectives should not be None here.")
                    case DefaultCollector(_):
                        obj_collection_dict = {key: [val]
                                               for key, val in zip(env.objectives.objectives_str_list, chosen_sum)}
                        obj_collection_dict["Step"] = [env.current_step]
                        chosen_obj_collector.append(obj_collection_dict)

            observations, rewards, done, _ = env.step(actions)

            for vehicle_id, reward in rewards.items():
                if vehicle_id in agents:
                    if vehicle_id in done:
                        previous_state = observations[vehicle_id]['previous_state']
                        next_state = observations[vehicle_id]['current_state']
                        handle_learning(vehicle_id, previous_state, next_state, reward)
                    else:
                        previous_state = observations[vehicle_id]['last_link_state']
                        next_state = observations[vehicle_id]['previous_state']
                        handle_learning(vehicle_id, previous_state, next_state, reward)
        env.close()
        if iteration != -1:
            logging.info("Iteration %s finished.", iteration)

    def create_agent(vehicle_id: str) -> None:
        """Method that creates a learning agent and puts it in the agents dictionary.

        Args:
            vehicle_id (str): vehicle id to identify the agent.
        """

        match config:
            case QLConfig(_):
                agents[vehicle_id] = QLAgent(action_space=env.action_space,
                                             exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.05),
                                             alpha=config.alpha,
                                             gamma=config.gamma)
            case PQLConfig(_):
                agents[vehicle_id] = PQLAgent(action_space=env.action_space,
                                              exploration_strategy=EpsilonGreedy(
                                                  initial_epsilon=0.05, min_epsilon=0.05),
                                              gamma=config.gamma)
            case _:
                raise RuntimeError(f"Config class not recognized.")

    def handle_learning(vehicle_id: str, origin_node: str, destination_node: str, reward: np.ndarray) -> None:
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
            action = env.get_action(origin_node, destination_node)
            match agents:
                case dict(QLAgent(_)):
                    assert isinstance(config, QLConfig)
                    obj = 0 if config.objective == "TravelTime" else 1
                    agents[vehicle_id].learn(action, origin_node, destination_node, reward[obj])
                case dict(PQLAgent(_)):
                    agents[vehicle_id].learn(action, origin_node, destination_node, reward)

        except Exception as exception:
            print(f"{vehicle_id = }")
            print(f"{observations = }")
            print(f"{rewards = }")
            raise Exception(exception).with_traceback(exception.__traceback__)

    def handle_communication(vehicle_id: str, state: str) -> None:
        """Method that retrieves CommDevs information if the C2I communication succeeds to update the agent's knowledge
        about the network.

        Args:
            vehicle_id (str): ID of the vehicle that will communicate with the CommDev.
            state (str): the state the CommDev is present.
        """
        comm_dev = env.get_comm_dev(state)
        if comm_dev.communication_success:
            expected_rewards = comm_dev.get_outgoing_links_expected_rewards()
            for link, expected_reward in expected_rewards.items():
                origin = env.get_link_origin(link)
                destination = env.get_link_destination(link)
                handle_learning(vehicle_id, origin, destination, expected_reward)

    # Run the simulation
    env = create_environment(config, graph_neighbours_dict)
    run(iteration)


def parse_args(command_line: str | None = None) -> NonLearnerConfig | QLConfig | PQLConfig:
    """Method that parse arguments coming from command line and returns a config with all the atributes necessary for a
    simulation run

    Args:
        command_line (str | None, optional): Command line arguments in a string. Defaults to None.

    Returns:
        NonLearnerConfig | QLConfig | PQLConfig: config structure with all parameters to be used in a simulation run
    """
    parser = create_parser()
    options = parser.parse_args(command_line)

    try:
        config: QLConfig | NonLearnerConfig | PQLConfig = options.func(options)
    except AttributeError:
        print("Wrong usage of script")
        parser.print_help()
        sys.exit(1)

    return config


def main(command_line=None):
    """Main script funcion that starts the running process.
    """
    config = parse_args(command_line)

    if config.nruns == 1:
        run_sim(config)
        return

    curr_date = datetime.now()
    if config.parallel:
        sys.setrecursionlimit(3000)
        with Pool(processes=os.cpu_count()) as pool:
            _ = [pool.apply_async(run_sim, args=(config, curr_date, it)) for it in range(config.nruns)]
            pool.close()
            pool.join()
        return

    for i in range(config.nruns):
        run_sim(config, curr_date, i)


if __name__ == '__main__':
    main()
