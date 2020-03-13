'''
Created Date: Wednesday, January 22nd 2020, 10:29:59 am
Author: Guilherme Dytz dos Santos
-----
Last Modified: Wednesday, March 4th 2020, 4:12 pm
Modified By: guilhermedytz
'''
# pylint: disable=fixme, line-too-long, invalid-name, missing-docstring
import sys
import os
from xml.dom import minidom
from datetime import datetime
from contextlib import contextmanager
import random as rd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import Environment
import traci
import sumolib

class SUMO(Environment):
    '''
    Create the environment as a MDP. The MDP is modeled as follows:
    * states represent nodes
    * actions represent the out links in each node
    * the reward of taking an action a (link) in state s (node) and going to a new state s' is the
      time spent on traveling on such a link multiplied by -1 (the lower the travel time the better)
    * the transitions between states are deterministic
    '''
    def __init__(self, cfg_file, port=8813, use_gui=False, time_before_learning=5000, max_veh=1000):

        super(SUMO, self).__init__()

        #check for SUMO's binaries
        if use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        #register SUMO/TraCI parameters
        self._cfg_file = cfg_file
        self._net_file = self._cfg_file[:self._cfg_file.rfind("/")+1] + minidom.parse(self._cfg_file).getElementsByTagName('net-file')[0].attributes['value'].value
        self._rou_file = self._cfg_file[:self._cfg_file.rfind("/")+1] + minidom.parse(self._cfg_file).getElementsByTagName('route-files')[0].attributes['value'].value
        self._port = port

        self.time_before_learning = time_before_learning
        self.max_veh = max_veh
        self.total_route_lengths = 0

        self.od_pair_set = set()
        self.od_pair_load = dict()
        self.od_pair_min = dict()

        #..............................

        #read the network file
        self._net = sumolib.net.readNet(self._net_file)

        self._env = {}
        for s in self._net.getNodes(): #current states (current nodes)
            d = {}
            for a in s.getOutgoing(): #actions (out links)
                d[a.getID()] = a.getToNode().getID() #resulting states (arriving nodes)
            self._env[s.getID()] = d

        self._env = {}
        for s in self._net.getNodes(): #current states (current nodes)
            for si in s.getIncoming():
                state = '%s:::%s'%(s.getID(), si.getID())
                d = {}
                for a in si.getOutgoing(): #actions (out links)
                    res_state = '%s:::%s'%(a.getToNode().getID(), a.getID())
                    d[a.getID()] = res_state #resulting states (arriving nodes)
                self._env[state] = d
            d = {}
            for a in s.getOutgoing(): #actions (out links)
                d[a.getID()] = a.getToNode().getID() #resulting states (arriving nodes)
            self._env[s.getID()] = d

        self.__create_vehicles()

    def get_all_states_id(self):
        states = self._net.getNodes()
        ids = [s.getID() for s in states]
        return ids

    def __create_vehicles(self):
        # set of all vehicles in the simulation
        # each element in _vehicles correspond to another in _agents
        # the SUMO's vehicles themselves are not stored, since the simulation
        # is recreated on each episode
        self._vehicles = {}

        #process all route entries
        R = {}

        # process all vehicle entries
        vehicles_parse = minidom.parse(self._rou_file).getElementsByTagName('vehicle')
        for v in vehicles_parse:
            #vehicle's ID
            vehID = v.getAttribute('id')

            # process the vehicle's route
            route = ''
            if v.hasAttribute('route'): # list of edges or route ID
                route = v.getAttribute('route')
                if route in R: # route ID
                    route = R[route]
            else: # child route tag
                route = v.getElementsByTagName('route')[0].getAttribute('edges')
                R[vehID] = route

            # origin and destination nodes
            origin = self.__get_edge_origin(route.split(' ')[0])
            destination = self.__get_edge_destination(route.split(' ')[-1])

            #update OD pairs
            od_pair = origin + destination
            if od_pair not in self.od_pair_set:
                self.od_pair_set.add(od_pair)
                self.total_route_lengths += len(route.split(' '))
            #depart
            depart = float(v.getAttribute('depart'))

            #vType
            vType = v.getAttribute('vType')

            # create the entry in the dictionary
            self._vehicles[vehID] = {
                'origin': origin,
                'destination': destination,
                'current_link': None,
                'previous_node': origin,

                # 'next_chosen': False,

                'desired_departure_time': int(depart), #desired departure time
                'departure_time': -1.0, #real departure time (as soon as the vehicle is no more waiting)
                'arrival_time': -1.0,
                'travel_time': -1.0,
                'time_last_link': -1.0,

                'route': [origin],
                'original_route': route.split(' '),

                'vType': vType
            }

        for route in R:
            od_pair = self.__get_edge_origin(R[route].split(' ')[0]) + self.__get_edge_destination(R[route].split(' ')[-1])
            self.od_pair_min.update({od_pair:int(len(R[route].split(' ')) / self.total_route_lengths * self.max_veh)})
            self.od_pair_load.update({od_pair:0})

    def get_vehicles_ID_list(self):
        #return a list with the vehicles' IDs
        return self._vehicles.keys()

    def get_vehicle_dict(self, vehID):
        # return the value in _vehicles corresponding to vehID
        return self._vehicles[vehID]

    def __get_edge_origin(self, edge_id):
        # return the FROM node ID of the edge edge_id
        return self._net.getEdge(edge_id).getFromNode().getID()

    def __get_edge_destination(self, edge_id):
        # return the TO node ID of the edge edge_id
        return self._net.getEdge(edge_id).getToNode().getID()

    #return an Edge instance from its ID
    def __get_action(self, ID):
        return self._net.getEdge(ID)

    #return a Node instance from its ID
    def __get_state(self, ID):
        return self._net.getNode(ID)

    #commands to be performed upon normal termination
    def __close_connection(self):
        traci.close()               #stop TraCI
        sys.stdout.flush()          #clear standard output

    def get_state_actions(self, state):
        return self._env[state].keys()

    def reset_episode(self):

        super(SUMO, self).reset_episode()

        #initialise TraCI
        traci.start([self._sumo_binary , "-c", self._cfg_file]) # SUMO 0.28

        #------------------------------------
        for vehID in self.get_vehicles_ID_list():
            self._vehicles[vehID]['current_link'] = None
            self._vehicles[vehID]['previous_node'] = self._vehicles[vehID]['origin']
            self._vehicles[vehID]['departure_time'] = -1.0
            self._vehicles[vehID]['arrival_time'] = -1.0
            self._vehicles[vehID]['travel_time'] = -1.0
            self._vehicles[vehID]['time_last_link'] = -1.0
            self._vehicles[vehID]['route'] = [self._vehicles[vehID]['origin']]
            self._vehicles[vehID]['initialized'] = False
            self._vehicles[vehID]['n_of_traversed_links'] = 0

    @contextmanager
    def redirected(self):
        saved_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        self._env = {}
        for s in self._net.getNodes(): #current states (current nodes)
            for si in s.getIncoming():
                state = '%s:::%s'%(s.getID(), si.getID())
                d = {}
                for a in si.getOutgoing(): #actions (out links)
                    res_state = '%s:::%s'%(a.getToNode().getID(), a.getID())
                    d[a.getID()] = res_state #resulting states (arriving nodes)
                self._env[state] = d
            d = {}
            for a in s.getOutgoing(): #actions (out links)
                d[a.getID()] = a.getToNode().getID() #resulting states (arriving nodes)
            self._env[s.getID()] = d
        # print (states and actions)
        for s in self._env.keys():
            print (s)
            for a in self._env[s]:
                print (a)#print '\t%s goes from %s to %s' % (a, self.__get_action(a).getFromNode().getID(), self.__get_action(a).getToNode().getID())

        # create the set of vehicles
        # self.__create_vehicles()
        yield
        sys.stdout = saved_stdout

    def run_episode(self, max_steps=-1, mv_avg_gap=100):
        # start = time.time()

        self._has_episode_ended = False
        self._episodes += 1
        self.reset_episode()
        travel_times = np.array([])
        travel_avg_df = pd.DataFrame({"Step":[], "Travel moving average times from arrived cars":[]})
        cars_over_5k = pd.DataFrame({"Step":[], "Number of arrived cars over 5k":[]})
        not_switched = True
        higher_count = 0
        total_count = 0
        start_time = datetime.now()
        log_path = os.getcwd() + '/log/sim_' + str(max_steps) + '_steps_' + start_time.strftime("%d-%m-%y_%H-%M")
        over_5k = log_path + '/over_5k'
        teleport = log_path + '/teleports.txt'
        try:
            os.mkdir(log_path)
        except OSError:
            print ("Creation of the directory %s failed" % log_path)
            traci.close()
            sys.exit()

        try:
            os.mkdir(over_5k)
        except OSError:
            print ("Creation of the directory %s failed" % over_5k)
            traci.close()
            sys.exit()

        for vehID in self.get_vehicles_ID_list():
            routeID = 'r_' + vehID
            if routeID not in traci.route.getIDList():
                _, action = self._agents[vehID].take_action()
                traci.route.add(routeID, [action])

        while traci.simulation.getTime() < max_steps:
            current_time = traci.simulation.getTime()
            vehicles_to_process_feedback = {}
            vehicles_to_process_act = {}
            count_over_5k = 0

            for vehID in traci.simulation.getStartingTeleportIDList():
                try:
                    with open(teleport, 'a') as txt_file:
                        teleport_str = "Vehicle " + vehID
                        teleport_str += " in link " + traci.vehicle.getRoadID(vehID)
                        teleport_str += " teleported in step " + str(current_time) + "\n"
                        txt_file.write(teleport_str)
                        txt_file.close()
                except IOError:
                    print("Unable to open " + teleport + " file")

            for vehID in traci.simulation.getArrivedIDList():
                self.__check_min_load(vehID)

                self._vehicles[vehID]["arrival_time"] = current_time
                self._vehicles[vehID]["travel_time"] = self._vehicles[vehID]["arrival_time"] - self._vehicles[vehID]["departure_time"]

                travel_times = np.append(travel_times, [self._vehicles[vehID]["travel_time"]])
                total_count += 1

                if self._vehicles[vehID]["travel_time"] > 5000:
                    higher_count += 1
                    count_over_5k += 1

                reward = current_time - self._vehicles[vehID]['time_last_link']
                reward *= -1

                if traci.simulation.getTime() > self.time_before_learning:
                    vehicles_to_process_feedback[vehID] = [
                        reward,
                        self.__get_edge_destination(self._vehicles[vehID]['current_link']),
                        self.__get_edge_origin(self._vehicles[vehID]['current_link']),
                        self._vehicles[vehID]['current_link']
                    ]

            if count_over_5k > 0:
                df = pd.DataFrame({"Step": [traci.simulation.getTime()],
                                   "Number of arrived cars over 5k": [count_over_5k]})
                cars_over_5k = cars_over_5k.append(df, ignore_index=True)

            for vehID in traci.simulation.getLoadedIDList():
                self._vehicles[vehID]['current_link'] = None
                self._vehicles[vehID]['previous_node'] = self._vehicles[vehID]['origin']
                self._vehicles[vehID]['departure_time'] = -1.0
                self._vehicles[vehID]['arrival_time'] = -1.0
                self._vehicles[vehID]['travel_time'] = -1.0
                self._vehicles[vehID]['time_last_link'] = -1.0
                self._vehicles[vehID]['route'] = [self._vehicles[vehID]['origin']]
                self._vehicles[vehID]['initialized'] = False
                self._vehicles[vehID]['n_of_traversed_links'] = 0
                od_pair = self._vehicles[vehID]['origin'] + self._vehicles[vehID]['destination']
                self.od_pair_load[od_pair] += 1
                routeID = 'r_' + vehID
                traci.vehicle.setRouteID(vehID, routeID)

            # departed vehicles (those that are entering the network)
            for vehID in traci.simulation.getDepartedIDList():
                self._vehicles[vehID]["departure_time"] = current_time

            for vehID in self.get_vehicles_ID_list(): # all vehicles
                # who have departed but not yet arrived
                if self._vehicles[vehID]["departure_time"] != -1.0 and self._vehicles[vehID]["arrival_time"] == -1.0:
                    road = traci.vehicle.getRoadID(vehID)
                    if road != self._vehicles[vehID]["current_link"] and self.__is_link(road): #but have just leaved a node
                        #update info of previous link
                        if self._vehicles[vehID]['time_last_link'] > -1.0:
                            reward = current_time - self._vehicles[vehID]['time_last_link']
                            reward *= -1

                            if traci.simulation.getTime() > self.time_before_learning:
                                vehicles_to_process_feedback[vehID] = [
                                    reward,
                                    self.__get_edge_destination(self._vehicles[vehID]['current_link']),
                                    self.__get_edge_origin(self._vehicles[vehID]['current_link']),
                                    self._vehicles[vehID]['current_link']
                                ]

                        self._vehicles[vehID]['time_last_link'] = current_time
                        self._vehicles[vehID]['travel_time'] = current_time - self._vehicles[vehID]['departure_time']

                        if self._vehicles[vehID]['travel_time'] > 5000:
                            filename = over_5k + '/' + vehID + '.txt'
                            try:
                                with open(filename, 'a') as txt_file:
                                    log_str = "time step " + str(traci.simulation.getTime()) + ": "
                                    log_str += "Current state is " + self._vehicles[vehID]['route'][-1] + ", "
                                    log_str += "took action " + self._vehicles[vehID]['current_link'] + ", "
                                    log_str += "with a reward of " + str(reward) + '\n'
                                    txt_file.write(log_str)
                                    txt_file.close()
                            except IOError:
                                print("Couldn't open file " + filename)

                        #update current_link
                        self._vehicles[vehID]['current_link'] = road
                        self._vehicles[vehID]['n_of_traversed_links'] += 1

                        #get the next node, and add it to the route
                        node = self.__get_edge_destination(self._vehicles[vehID]["current_link"])
                        self._vehicles[vehID]['route'].append(self.__get_edge_destination(self._vehicles[vehID]['current_link']))

                        if node != self._vehicles[vehID]['destination']:
                            if traci.simulation.getTime() > self.time_before_learning:
                                outgoing = self._net.getEdge(self._vehicles[vehID]['current_link']).getOutgoing()
                                possible_actions = [edge.getID() for edge in outgoing if len(edge.getOutgoing(
                                )) > 0 or self.__get_edge_destination(edge.getID()) == self._vehicles[vehID]['destination']]
                                # for edge in self._net.getEdge(self._vehicles[vehID]['current_link']).getOutgoing():
                                #     if len(edge.getOutgoing()) > 0:
                                #         possible_actions.append(edge.getID())
                                #     elif :
                                #         possible_actions.append(edge.getID())
                                vehicles_to_process_act[vehID] = [
                                    node, #next state
                                    possible_actions #available actions
                                ]
                            else:
                                vehicles_to_process_act[vehID] = [
                                    node, #next state
                                    [self._vehicles[vehID]['original_route'][len(self._vehicles[vehID]['route']) - 1]] #available actions
                                ]


            self.__process_vehicles_feedback(vehicles_to_process_feedback)
            self.__process_vehicles_act(vehicles_to_process_act, current_time)

            if traci.simulation.getTime() > (max_steps / 2) and not_switched:
                for vehID in traci.vehicle.getIDList():
                    self._agents[vehID].switch_epsilon(0)
                not_switched = False

            step = traci.simulation.getTime()
            if step % mv_avg_gap == 0 and step > 0:
                df = pd.DataFrame({"Step": [step],
                                   "Travel moving average times from arrived cars": [travel_times.mean()]})
                travel_avg_df = travel_avg_df.append(df, ignore_index=True)
                travel_times = np.array([])

            traci.simulationStep()


        traci.close()
        try:
            with open('sims_log.txt', 'a') as logfile:
                end_time = datetime.now()
                log_str = "-----------------------------------------------\n"
                log_str += "Simulation with " + str(max_steps) + " steps run in " + start_time.strftime("%d/%m/%y") + "\n"
                log_str += "Start time: " + start_time.strftime("%H:%M") + "\n"
                log_str += "End time: " + end_time.strftime("%H:%M") + "\n"
                log_str += "Total trips ended: " + str(total_count) + "\n"
                log_str += "Trips that ended with more than 5k steps: " + str(higher_count) + "\n"
                log_str += "Percentage (higher / total): " + "{:.2f} %\n\n".format(higher_count / total_count * 100)

                logfile.write(log_str)
                logfile.close()
        except IOError:
            print("Unable to open simulations log file")
        travel_avg_df.plot(kind="scatter", x="Step", y="Travel moving average times from arrived cars")
        plt.show()
        # start_time = datetime.now()
        sim_name = 'csv/MovingAverage/sim_' + str(max_steps) + '_steps_' + start_time.strftime("%d-%m-%y_%H-%M") + ".csv"
        travel_avg_df.to_csv(sim_name, index=False)

        cars_over_5k.plot(kind="scatter", x="Step", y="Number of arrived cars over 5k")
        plt.show()
        plot_name = 'csv/CarsOver5k/sim_' + str(max_steps) + '_steps_' + start_time.strftime("%d-%m-%y_%H-%M") + ".csv"
        cars_over_5k.to_csv(plot_name, index=False)

    def __process_vehicles_feedback(self, vehicles):
        # feedback_last
        for vehID in vehicles.keys():
            self._agents[vehID].process_feedback(vehicles[vehID][0], vehicles[vehID][1], vehicles[vehID][2], vehicles[vehID][3])

    def __process_vehicles_act(self, vehicles, current_time):

        # act_last
        for vehID in vehicles.keys():
            #~ print vehID,  vehicles[vehID][1]
            _, action = self._agents[vehID].take_action(vehicles[vehID][0], vehicles[vehID][1])
            #~ print vehID, action
            #print "%s is in state %s and chosen action %s among %s" % (vehID, vehicles[vehID][0], action, vehicles[vehID][1])

            if not vehicles[vehID][1]:
                traci.vehicle.remove(vehID, traci.constants.REMOVE_ARRIVED)
                self._vehicles[vehID]["arrival_time"] = current_time
                self._vehicles[vehID]["travel_time"] = self._vehicles[vehID]["arrival_time"] - self._vehicles[vehID]["departure_time"]
                continue

            #update route
            cur_route = list(traci.vehicle.getRoute(vehID))
            #~ print 'route', traci.route.getEdges('R-%s'%vehID)
            #~ print vehID, traci.vehicle.getRoute(vehID)
            cur_route.append(action)
            #~ print 'current ', vehID, self._vehicles[vehID]['current_link']

            # remove traversed links from the route
            # (this is necessary because otherwise the driver will try
            # to reach the first link of such route from its current link)
            cur_route = cur_route[self._vehicles[vehID]['n_of_traversed_links']-1:]

            #~ print vehID, cur_route
            traci.vehicle.setRoute(vehID, cur_route)

    def __is_link(self, edge_id):
        try:
            _ = self._net.getEdge(edge_id)
            return True
        except NameError:
            return False

    def run_step(self):
        raise Exception('run_step is not available in %s class' % self)

    def has_episode_ended(self):
        return self._has_episode_ended

    def __calc_reward(self, state, action, new_state):
        raise Exception('__calc_reward is not available in %s class' % self)

    def get_starting_edge_value(self, edge_id):
        origin = self._net.getNode(self.__get_edge_origin(edge_id))
        dest = self._net.getNode(self.__get_edge_destination(edge_id))
        dist = np.linalg.norm(np.array(dest.getCoord()) - np.array(origin.getCoord()))
        speed = self._net.getEdge(edge_id).getSpeed()
        base_value = dist / speed
        return - base_value + rd.uniform(0, - base_value)

    def __check_min_load(self, vehID):
        od_pair = self._vehicles[vehID]['origin'] + self._vehicles[vehID]['destination']
        self.od_pair_load[od_pair] -= 1
        if self.od_pair_load[od_pair] < self.od_pair_min[od_pair]:
            routeID = 'r_' + vehID
            if routeID not in traci.route.getIDList():
                _, action = self._agents[vehID].take_action()
                traci.route.add(routeID, [action])
            traci.vehicle.add(vehID, routeID)





# SAVE CSV
# csv_file = "csv/Q-Table/Over5k/" + vehID + '_' + str(traci.simulation.getTime()) +  ".csv"

                # QTable = self._agents[vehID].get_Q_table()
                # try:
                #     with open(csv_file, 'w') as csvfile:
                #         csv_columns = [edge for edge in traci.edge.getIDList() if edge.find(':') == -1]
                #         csv_columns.insert(0, 'State')
                #         writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                #         writer.writeheader()
                #         for key in QTable.keys():
                #             row = QTable[key].copy()
                #             row['State'] = key
                #             writer.writerow(row)
                # except IOError:
                #     print("I/O Error")
