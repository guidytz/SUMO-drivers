'''
Created Date: Wednesday, January 22nd 2020, 10:51 am
Author: Guilherme Dytz dos Santos
-----
Last Modified: Saturday, March 14th 2020, 2:47 pm
Modified By: guilhermedytz
'''
# pylint: disable=fixme, line-too-long, invalid-name, missing-docstring
import sys
import argparse
from environment.sumo import SUMO
from agent.q_learning import QLearner
from exploration.epsilon_greedy import EpsilonGreedy


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
    parse.add_argument("-m", "--mav", action="store",type=int, dest="mav", default=100,
                       help="Moving gap size (default = 100 steps)")
    parse.add_argument("--no-c2i", action="store_false", dest="c2i", default=True,
                       help="Toggle not to use car-to-infrastructure communication")
    parse.add_argument("-r", "--success-rate", action="store", type=float, dest="sr", default=1,
                       help="Communication success rate (default = 1)")

    options = parse.parse_args()
    if not options.cfgfile:
        print('Wrong usage of script!')
        print()
        parse.print_help()
        sys.exit()

    env = SUMO(options.cfgfile, use_gui=options.gui, time_before_learning=options.wait_learn, max_veh=options.demand)

    agents = list()
    for veh in env.get_vehicles_ID_list():
        veh_dict = env.get_vehicle_dict(veh)
        exp = EpsilonGreedy(0.05, 0, -1)
        agent = QLearner(veh, env, veh_dict['origin'], veh_dict['destination'], 0.5, 0.9, exp)
        agents.append(agent)

    env.register_agents(agents)
    env.update_c2i_params(options.c2i, options.sr)

    env.run_episode(options.steps, options.mav)

