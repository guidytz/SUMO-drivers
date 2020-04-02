'''
Created Date: Wednesday, January 22nd 2020, 10:29:59 am
Author: Guilherme Dytz dos Santos
-----
Last Modified: Wednesday, April 1st 2020, 11:04 am
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
import math

class SUMO(Environment):
    """
    Create the environment as a MDP. The MDP is modeled as follows:
    * states represent nodes
    * actions represent the out links in each node
    * the reward of taking an action a (link) in state s (node) and going to a new state s' is the
      time spent on traveling on such a link multiplied by -1 (the lower the travel time the better)
    * the transitions between states are deterministic
    """
    def __init__(self, cfg_file, port=8813, use_gui=False, time_before_learning=5000, max_veh=1000, max_queue_val=30, class_interval=200, top_class_value=5000):
        self.__flags = {
            'C2I': False,
            'over5k_log': False,
            'teleport_log': False,
            'plot': True,
            'plot_over5k': False
        }
        
        rd.seed(datetime.now())


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
        self.__vehicles_to_process_feedback = {}
        self.__vehicles_to_process_act = {}

        self.time_before_learning = time_before_learning
        self.max_veh = max_veh
        self.total_route_lengths = 0

        self.od_pair_set = set()
        self.od_pair_load = dict()
        self.od_pair_min = dict()
        self.comm_devices = dict()
        self.max_queue = max_queue_val
        self.log_sample = list()
        self.class_interval = class_interval
        self.top_class_value = top_class_value
        self.classifier = self.__create_data_classifier(self.class_interval, self.top_class_value)

        #..............................

        #read the network file
        self._net = sumolib.net.readNet(self._net_file)

        # create structure to handle C2I communication
        for edge in self._net.getEdges():
            self.comm_devices[edge.getID()] = list()

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
        """
        Sets up the structure that holds all information necessary for vehicle handling in
        the simulation
        """
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

    def run_episode(self, max_steps=-1, mv_avg_gap=100):
        self._has_episode_ended = False
        self._episodes += 1
        self.reset_episode()
        self.travel_times = np.array([])
        travel_avg_df = pd.DataFrame({"Step":[], "Travel moving average times from arrived cars":[]})
        cars_over_5k = pd.DataFrame({"Step":[], "Number of arrived cars over 5k":[]}) if self.__flags['plot_over5k'] else None
        not_switched = True
        higher_count = 0
        total_count = 0
        start_time = datetime.now()
        log_path = os.getcwd() + '/log/sim_' + str(max_steps) + '_steps_' + start_time.strftime("%d-%m-%y_%H-%M")
        over_5k = log_path + '/over_5k'
        self.sample_path = log_path + '/sample'
        teleport = log_path + '/teleports.txt'
        self.trips_per_od = {od : 0 for od in self.od_pair_set}

        self.__make_log_folder(log_path)
        self.__make_log_folder(over_5k)
        self.__make_log_folder(self.sample_path)

        for vehID in self.get_vehicles_ID_list():
            routeID = 'r_' + vehID
            if routeID not in traci.route.getIDList():
                _, action = self._agents[vehID].take_action()
                traci.route.add(routeID, [action])

        while traci.simulation.getTime() < max_steps:
            current_time = traci.simulation.getTime()
            self.__vehicles_to_process_feedback = {}
            self.__vehicles_to_process_act = {}
            higher_per_step = 0

            if self.__flags['teleport_log'] : self.__update_teleport_log(teleport, current_time)

            [total_count, higher_per_step] = self.__process_arrived(current_time, total_count, higher_per_step)

            if higher_per_step > 0 and self.__flags['plot_over5k']:
                df = pd.DataFrame({"Step": [traci.simulation.getTime()],
                                   "Number of arrived cars over 5k": [higher_per_step]})
                cars_over_5k = cars_over_5k.append(df, ignore_index=True)

            if traci.simulation.getTime() == self.time_before_learning:
                self.log_sample = self.__sample_log(self.sample_path)

            self.__update_loaded_info()

            # departed vehicles (those that are entering the network)
            self.__update_departed_info(current_time)
            self.__process_all(current_time, over_5k)
            self.__process_vehicles_feedback(self.__vehicles_to_process_feedback)
            self.__process_vehicles_act(self.__vehicles_to_process_act, current_time)

            if traci.simulation.getTime() > (max_steps / 2) and not_switched:
                for vehID in traci.vehicle.getIDList():
                    self._agents[vehID].switch_epsilon(0)
                not_switched = False

            step = traci.simulation.getTime()
            if step % mv_avg_gap == 0 and step > 0:
                df = pd.DataFrame({"Step": [step],
                                   "Travel moving average times from arrived cars": [self.travel_times.mean()]})
                travel_avg_df = travel_avg_df.append(df, ignore_index=True)
                self.travel_times = np.array([])

            higher_count += higher_per_step
            traci.simulationStep()

        traci.close()
        self.__write_sim_logfile(max_steps, start_time, total_count, higher_count)
        
        sort = {key : self.trips_per_od[key] for key in sorted(self.trips_per_od)}
        frame = {"OD Pair": list(sort.keys()), "Number of Trips Ended": list(sort.values())}
        trips_dataframe = pd.DataFrame(frame)
        class_df = self.__create_class_dataframe()
        
        if self.__flags['plot']:
            travel_avg_df.plot(kind="scatter", x="Step", y="Travel moving average times from arrived cars")
            plt.xlabel("Step")
            plt.ylabel("Travel Moving Average Times From Arrived Cars")
             
            plt.figure(1)
            trips_dataframe.plot(x="OD Pair", y="Number of Trips Ended", rot=0, figsize=(15, 5), kind="bar")
            plt.xlabel("OD Pair")
            plt.ylabel("Number of Trips Ended")
            
            plt.figure(2)
            class_df.plot(x="Interval", y="Trips Ended Within the Interval",kind='bar', figsize=(15, 8))
            plt.xlabel("Interval")
            plt.ylabel("Trips Ended Within the Interval")

            plt.show()

        sim_name = 'csv/ClassDivision/sim_' + str(max_steps) + '_steps_' + start_time.strftime("%d-%m-%y_%H-%M")
        sim_name += "_withC2I.csv" if self.__flags['C2I'] else '.csv'
        class_df.to_csv(sim_name, index=False)

        sim_name = 'csv/TripsPerOD/sim_' + str(max_steps) + '_steps_' + start_time.strftime("%d-%m-%y_%H-%M")
        sim_name += "_withC2I.csv" if self.__flags['C2I'] else '.csv'
        trips_dataframe.to_csv(sim_name, index=False)
        
        sim_name = 'csv/MovingAverage/sim_' + str(max_steps) + '_steps_' + start_time.strftime("%d-%m-%y_%H-%M")
        sim_name += "_withC2I.csv" if self.__flags['C2I'] else '.csv'
        travel_avg_df.to_csv(sim_name)

        if self.__flags['plot_over5k']:
            cars_over_5k.plot(kind="scatter", x="Step", y="Number of arrived cars over 5k")
            plt.show()

            plot_name = 'csv/CarsOver5k/sim_' + str(max_steps) + '_steps_' + start_time.strftime("%d-%m-%y_%H-%M") + ".csv"
            cars_over_5k.to_csv(plot_name, index=False)

        print()

    def __process_vehicles_feedback(self, vehicles):
        # feedback_last
        for vehID in vehicles.keys():
            self._agents[vehID].process_feedback(vehicles[vehID][0], vehicles[vehID][1], vehicles[vehID][2], vehicles[vehID][3])
            self._agents[vehID].process_feedback(vehicles[vehID][0], vehicles[vehID][1], vehicles[vehID][2], vehicles[vehID][3], 1)

    def __process_vehicles_act(self, vehicles, current_time):

        # act_last
        for vehID in vehicles.keys():
            self.__upd_c2i_info(vehID, vehicles[vehID][0])

            use_C2I = 1 if self.__flags['C2I'] else 0

            _, action = self._agents[vehID].take_action(vehicles[vehID][0], vehicles[vehID][1], use_C2I)

            if not vehicles[vehID][1]:
                traci.vehicle.remove(vehID, traci.constants.REMOVE_ARRIVED)
                self._vehicles[vehID]["arrival_time"] = current_time
                self._vehicles[vehID]["travel_time"] = self._vehicles[vehID]["arrival_time"] - self._vehicles[vehID]["departure_time"]
                continue

            #update route
            cur_route = list(traci.vehicle.getRoute(vehID))
            cur_route.append(action)

            # remove traversed links from the route
            # (this is necessary because otherwise the driver will try
            # to reach the first link of such route from its current link)
            cur_route = cur_route[self._vehicles[vehID]['n_of_traversed_links']-1:]

            traci.vehicle.setRoute(vehID, cur_route)

    def __is_link(self, edge_id):
        try:
            _ = self._net.getEdge(edge_id)
            return True
        except:
            return False

    def run_step(self):
        raise Exception('run_step is not available in %s class' % self)

    def has_episode_ended(self):
        return self._has_episode_ended

    def __calc_reward(self, state, action, new_state):
        raise Exception('__calc_reward is not available in %s class' % self)

    def get_starting_edge_value(self, edge_id):
        # origin = self._net.getNode(self.__get_edge_origin(edge_id))
        # dest = self._net.getNode(self.__get_edge_destination(edge_id))
        # dist = np.linalg.norm(np.array(dest.getCoord()) - np.array(origin.getCoord()))
        # speed = self._net.getEdge(edge_id).getSpeed()
        # base_value = dist / speed
        # return - base_value + rd.uniform(0, - base_value)
        return 0

    def __check_min_load(self, vehID):
        od_pair = self._vehicles[vehID]['origin'] + self._vehicles[vehID]['destination']
        self.od_pair_load[od_pair] -= 1
        if self.od_pair_load[od_pair] < self.od_pair_min[od_pair]:
            routeID = 'r_' + vehID
            if routeID not in traci.route.getIDList():
                _, action = self._agents[vehID].take_action()
                traci.route.add(routeID, [action])
            traci.vehicle.add(vehID, routeID)

    def __update_loaded_info(self):
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

    def __update_departed_info(self, current_time):
        for vehID in traci.simulation.getDepartedIDList():
                self._vehicles[vehID]["departure_time"] = current_time

    def __process_arrived(self, current_time, total_count, higher_per_step):
        for vehID in traci.simulation.getArrivedIDList():
                self.__check_min_load(vehID)

                self._vehicles[vehID]["arrival_time"] = current_time
                self._vehicles[vehID]["travel_time"] = self._vehicles[vehID]["arrival_time"] - self._vehicles[vehID]["departure_time"]

                if self._vehicles[vehID]["travel_time"] < 5000:
                    self.travel_times = np.append(self.travel_times, [self._vehicles[vehID]["travel_time"]])
                total_count += 1

                if self._vehicles[vehID]["travel_time"] >= 5000 : higher_per_step += 1

                reward = current_time - self._vehicles[vehID]['time_last_link']
                reward *= -1

                if self._vehicles[vehID]["route"][-1] == self._vehicles[vehID]["destination"] and traci.simulation.getTime() >= self.time_before_learning:
                    od_pair = self._vehicles[vehID]["origin"] + self._vehicles[vehID]["destination"] 
                    self.trips_per_od[od_pair] += 1
                    index = math.floor(self._vehicles[vehID]["travel_time"] / self.class_interval)
                    if index >= len(self.classifier):
                        self.classifier[-1] += 1
                    else:
                        self.classifier[index] += 1

                if traci.simulation.getTime() > self.time_before_learning:
                    self.__vehicles_to_process_feedback[vehID] = [
                        reward,
                        self.__get_edge_destination(self._vehicles[vehID]['current_link']),
                        self.__get_edge_origin(self._vehicles[vehID]['current_link']),
                        self._vehicles[vehID]['current_link']
                    ]

                if vehID in self.log_sample and traci.simulation.getTime() > self.time_before_learning:
                    od_pair = self._vehicles[vehID]['origin'] + self._vehicles[vehID]['destination']
                    path = self.sample_path + "/" + str(od_pair)
                    self.__write_veh_log(path, vehID, reward, True)
        
        return [total_count, higher_per_step]

    def __process_all(self, current_time, over5k_path):
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
                                self.__update_inf_value(self._vehicles[vehID]['current_link'], reward)
                                self.__vehicles_to_process_feedback[vehID] = [
                                    reward,
                                    self.__get_edge_destination(self._vehicles[vehID]['current_link']),
                                    self.__get_edge_origin(self._vehicles[vehID]['current_link']),
                                    self._vehicles[vehID]['current_link']
                                ]

                            if self._vehicles[vehID]['travel_time'] > 5000 and self.__flags['over5k_log']:
                                self.__write_veh_log(over5k_path, vehID, reward)

                            if traci.simulation.getTime() > self.time_before_learning:
                                if vehID in self.log_sample:
                                    od_pair = self._vehicles[vehID]['origin'] + self._vehicles[vehID]['destination']
                                    path = self.sample_path + "/" + str(od_pair)
                                    self.__write_veh_log(path, vehID, reward)


                        self._vehicles[vehID]['time_last_link'] = current_time
                        self._vehicles[vehID]['travel_time'] = current_time - self._vehicles[vehID]['departure_time']

                        #update current_link
                        self._vehicles[vehID]['current_link'] = road
                        self._vehicles[vehID]['n_of_traversed_links'] += 1

                        #get the next node, and add it to the route
                        node = self.__get_edge_destination(self._vehicles[vehID]["current_link"])
                        self._vehicles[vehID]['route'].append(self.__get_edge_destination(self._vehicles[vehID]['current_link']))

                        if node != self._vehicles[vehID]['destination']:
                            if traci.simulation.getTime() > self.time_before_learning:
                                outgoing = self._net.getEdge(self._vehicles[vehID]['current_link']).getOutgoing()
                                possible_actions = list()
                                for edge in outgoing:
                                    if len(edge.getOutgoing()) > 0 or self.__get_edge_destination(edge.getID()) == self._vehicles[vehID]['destination']:
                                        possible_actions.append(edge.getID())
                                self.__vehicles_to_process_act[vehID] = [
                                    node, #next state
                                    possible_actions #available actions
                                ]
                            else:
                                self.__vehicles_to_process_act[vehID] = [
                                    node, #next state
                                    [self._vehicles[vehID]['original_route'][len(self._vehicles[vehID]['route']) - 1]] #available actions
                                ]

    def __update_inf_value(self, link_id, travel_time):
        self.comm_devices[link_id].append(travel_time)
        if len(self.comm_devices[link_id]) > self.max_queue:
            self.comm_devices[link_id].pop(0)
    
    def __upd_c2i_info(self, vehID, node):
        state = self._net.getNode(node)
        for edge in state.getOutgoing():
            edge_id = edge.getID()
            if len(self.comm_devices[edge_id]) > 0:
                possible_reward = np.array(self.comm_devices[edge.getID()]).mean()
                origin = self.__get_edge_origin(edge_id)
                destination = self.__get_edge_destination(edge_id)
                self._agents[vehID].process_feedback(possible_reward, destination, origin, edge_id, 1)

    def __make_log_folder(self, folder_name):
        try:
            os.mkdir(folder_name)
        except OSError:
            print ("Creation of the directory %s failed" % folder_name)
            traci.close()
            sys.exit()
    
    def __update_teleport_log(self, path, current_time):
        for vehID in traci.simulation.getStartingTeleportIDList():
                try:
                    with open(path, 'a') as txt_file:
                        teleport_str = "Vehicle " + vehID
                        teleport_str += " in link " + traci.vehicle.getRoadID(vehID)
                        teleport_str += " teleported in step " + str(current_time) + "\n"
                        txt_file.write(teleport_str)
                        txt_file.close()
                except IOError:
                    print("Unable to open " + path + " file")

    def __write_veh_log(self, path, vehID, reward, trip_end=False):
        filename = path + '/' + vehID + '.txt'
        try:
            with open(filename, 'a') as txt_file:
                c2i = 1 if self.__flags['C2I'] else 0
                rand_key = ''
                log_str = "time step " + str(traci.simulation.getTime()) + ": "
                log_str += "Current state is " + self._vehicles[vehID]['route'][-1] + ", "
                log_str += "took action " + self._vehicles[vehID]['current_link']
                log_str += ", "
                log_str += "with a reward of " + str(reward) + "  "
                QTable = self._agents[vehID].get_Q_table()
                if trip_end:
                    log_str += "\nTrip ended with travel time " + str(self._vehicles[vehID]['travel_time'])
                    log_str += "\n\n" 
                else:
                    max_val = max(QTable[c2i][self._vehicles[vehID]['route'][-2]].values())
                    for key, val in QTable[c2i][self._vehicles[vehID]['route'][-2]].items():
                        if val == max_val:
                            rand_key = key

                    if max_val == 0:
                        rand_key = self._vehicles[vehID]['current_link']
                    log_str += "\nNormal: " + str(QTable[0][self._vehicles[vehID]['route'][-1]]) + "  "
                    log_str += "\n   C2I: " + str(QTable[1][self._vehicles[vehID]['route'][-1]]) + "\n"
                log_str = ('' if rand_key == self._vehicles[vehID]['current_link'] else '*') + log_str
                txt_file.write(log_str)
                txt_file.close()
        except IOError:
            print("Couldn't open file " + filename)

    def __write_sim_logfile(self, total_steps, start_time, total_count, higher_count):
        try:
            with open('log/sims_log.txt', 'a') as logfile:
                end_time = datetime.now()
                log_str = "-----------------------------------------------\n"
                log_str += "Simulation with " + str(total_steps) + " steps run in " + start_time.strftime("%d/%m/%y") + "\n"
                log_str += "C2I was used: "
                log_str += "yes" if self.__flags['C2I'] else "no"
                log_str += "\n"
                log_str += "Start time: " + start_time.strftime("%H:%M") + "\n"
                log_str += "End time: " + end_time.strftime("%H:%M") + "\n"
                log_str += "Total trips ended: " + str(total_count) + "\n"
                log_str += "Trips that ended with more than 5k steps: " + str(higher_count) + "\n"
                log_str += "Percentage (higher / total): " + "{:.2f} %\n\n".format(higher_count / total_count * 100)

                logfile.write(log_str)
                logfile.close()
        except IOError:
            print("Unable to open simulations log file")

    def __sample_log(self, sample_path):
        all_veh = traci.vehicle.getIDList()
        od_sample = {od:list() for od in self.od_pair_set}
        sample = list()
        for veh in all_veh:
            od_pair = self._vehicles[veh]['origin'] + self._vehicles[veh]['destination']
            od_sample[od_pair].append(veh)

        for od_pair in od_sample.keys():
            sample.extend(rd.sample(od_sample[od_pair], 5))
            od_path = sample_path + '/' + str(od_pair)
            self.__make_log_folder(od_path)
        
        print("Sample size:",len(sample))

        return sample

    def __create_data_classifier(self, interval, max):
        step = 0
        classifier = list()
        while step <= max:
            classifier.append(0)
            step += interval

        return classifier

    def __create_class_dataframe(self):
        begin = 0
        class_dataframe = pd.DataFrame({"Interval":[], "Trips Ended Within the Interval":[]})
        while begin < self.top_class_value:
            index = begin // self.class_interval
            interval_name = str(begin) + ' - '
            begin += self.class_interval - 1
            interval_name += str(begin)
            begin += 1
            aux_df = pd.DataFrame({"Interval":[interval_name], "Trips Ended Within the Interval": [self.classifier[index]]})
            class_dataframe = class_dataframe.append(aux_df, ignore_index=True)

        aux_df = pd.DataFrame({"Interval":[str(self.top_class_value) + " or more"], "Trips Ended Within the Interval": [self.classifier[-1]]})
        class_dataframe = class_dataframe.append(aux_df, ignore_index=True)
            
        return class_dataframe






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
