import sys
import argparse
from typing import Dict
from sumo_ql.environment.sumo_environment import SumoEnvironment
from sumo_ql.agent.q_learning import QLAgent
from sumo_ql.exploration.epsilon_greedy import EpsilonGreedy


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

    env = SumoEnvironment(sumocfg_file=options.cfgfile,
                          simulation_time=options.steps,
                          max_vehicles=options.demand,
                          right_arrival_bonus=options.bonus,
                          wrong_arrival_penalty=options.penalty,
                          communication_success_rate=options.comm_succ_rate,
                          max_comm_dev_queue_size=options.queue_size,
                          steps_to_populate=options.wait_learn,
                          use_gui=options.gui)

    observations = env.reset()
    agents: Dict[str, QLAgent] = dict()
    done = {'__all__': False}
    while not done['__all__']:
        actions = dict()
        for vehicle_id in observations:
            if observations[vehicle_id]['reinserted']:
                if vehicle_id not in agents:
                    agents[vehicle_id] = QLAgent(action_space=env.action_space,
                                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.05,
                                                                                    min_epsilon=0.05))

        for vehicle_id in observations:
            if observations[vehicle_id]['ready_to_act'] and vehicle_id in agents:
                commDev = env.get_comm_dev(observations[vehicle_id]['current_state'])
                if commDev.communication_success:
                    expected_rewards = commDev.get_outgoing_links_expected_rewards()
                    for link, expected_reward in expected_rewards.items():
                        origin = env.get_link_origin(link)
                        destination = env.get_link_destination(link)
                        action = env.get_action(origin, destination)
                        agents[vehicle_id].learn(action, origin, destination, expected_reward)
                actions[vehicle_id] = agents[vehicle_id].act(observations[vehicle_id]['current_state'],
                                                             observations[vehicle_id]['available_actions'])
        observations, rewards, done, _ = env.step(actions)

        for vehicle_id, reward in rewards.items():
            if vehicle_id in agents:
                if vehicle_id in done:
                    try:
                        previous_state = observations[vehicle_id]['previous_state']
                        next_state = observations[vehicle_id]['current_state']
                        action = env.get_action(previous_state, next_state)
                        agents[vehicle_id].learn(action, previous_state, next_state, reward)
                    except Exception as exception:
                        print(f"{vehicle_id = }")
                        print(f"{observations = }")
                        print(f"{rewards = }")
                        raise Exception(exception).with_traceback(exception.__traceback__)
                else:
                    try:
                        previous_state = observations[vehicle_id]['last_link_state']
                        next_state = observations[vehicle_id]['previous_state']
                        action = env.get_action(previous_state, next_state)
                        agents[vehicle_id].learn(action, previous_state, next_state, reward)
                    except Exception as exception:
                        print(f"{vehicle_id = }")
                        print(f"{observations = }")
                        print(f"{rewards = }")
                        raise Exception(exception).with_traceback(exception.__traceback__)
    env.close()
