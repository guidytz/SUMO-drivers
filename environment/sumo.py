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
import traci.constants as tc
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
            'debug': False,
            'over5k_log': False,
            'teleport_log': False,
            'plot': False,
            'plot_over5k': False,
            'sample_log': False
        }
        
        rd.seed(datetime.now())

        super(SUMO, self).__init__()

        #check for SUMO's binaries
        if use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        #register SUMO/TraCI parameters
        self.__cfg_file = cfg_file
        self.__net_file = (self.__cfg_file[:self.__cfg_file.rfind("/")+1] + 
                          minidom.parse(self.__cfg_file).getElementsByTagName('net-file')[0].attributes['value'].value)
        self.__rou_file = (self.__cfg_file[:self.__cfg_file.rfind("/")+1] + 
                        minidom.parse(self.__cfg_file).getElementsByTagName('route-files')[0].attributes['value'].value)
        self.__port = port
        self.__vehicles_to_process_feedback = {}
        self.__vehicles_to_process_act = {}

        self.__time_before_learning = time_before_learning
        self.__max_veh = max_veh
        self.__total_route_dist = 0

        self.__od_pair_set = set()
        self.__od_pair_load = dict()
        self.__od_pair_min = dict()
        self.__comm_dev = dict()
        self.__comm_succ_rate = 1
        self.__max_queue = max_queue_val
        self.__log_sample = self.__class_interval = self.__top_class_value = self.__classifier = None
        if self.__flags['debug']:
            self.__log_sample = list()
            self.__class_interval = class_interval
            self.__top_class_value = top_class_value
            self.__classifier = self.__create_data_classifier(self.__class_interval, self.__top_class_value)

        #..............................

        #read the network file
        self.__net = sumolib.net.readNet(self.__net_file)

        # create structure to handle C2I communication
        for edge in self.__net.getEdges():
            self.__comm_dev[edge.getID()] = list()

        self._env = {}
        for s in self.__net.getNodes(): #current states (current nodes)
            d = {}
            for a in s.getOutgoing(): #actions (out links)
                d[a.getID()] = a.getToNode().getID() #resulting states (arriving nodes)
            self._env[s.getID()] = d

        self._env = {}
        for s in self.__net.getNodes(): #current states (current nodes)
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
        states = self.__net.getNodes()
        ids = [s.getID() for s in states]
        return ids

    def __create_vehicles(self):
        """
        Sets up the structure that holds all information necessary for vehicle handling in
        the simulation
        """
        self.__vehicles = {}

        #process all route entries
        R = {}

        # process all vehicle entries
        vehicles_parse = minidom.parse(self.__rou_file).getElementsByTagName('vehicle')
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
            if od_pair not in self.__od_pair_set:
                self.__od_pair_set.add(od_pair)
                o = np.array(self.__net.getNode(origin).getCoord())
                d = np.array(self.__net.getNode(destination).getCoord())
                self.__total_route_dist += np.linalg.norm(o - d)
            #depart
            depart = float(v.getAttribute('depart'))

            #vType
            vType = v.getAttribute('vType')

            # create the entry in the dictionary
            self.__vehicles[vehID] = {
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
            origin = self.__get_edge_origin(R[route].split(' ')[0])
            destination = self.__get_edge_destination(R[route].split(' ')[-1])
            od_pair = f"{origin}{destination}"
            o = np.array(self.__net.getNode(origin).getCoord())
            d = np.array(self.__net.getNode(destination).getCoord())
            self.__od_pair_min.update({od_pair:math.ceil(np.linalg.norm(o - d) / self.__total_route_dist * self.__max_veh)})
            self.__od_pair_load.update({od_pair:0})

    def get_vehicles_ID_list(self):
        #return a list with the vehicles' IDs
        return self.__vehicles.keys()

    def get_vehicle_dict(self, vehID):
        # return the value in __vehicles corresponding to vehID
        return self.__vehicles[vehID]

    def __get_edge_origin(self, edge_id):
        # return the FROM node ID of the edge edge_id
        return self.__net.getEdge(edge_id).getFromNode().getID()

    def __get_edge_destination(self, edge_id):
        # return the TO node ID of the edge edge_id
        return self.__net.getEdge(edge_id).getToNode().getID()

    #return an Edge instance from its ID
    def __get_action(self, ID):
        return self.__net.getEdge(ID)

    #return a Node instance from its ID
    def __get_state(self, ID):
        return self.__net.getNode(ID)

    #commands to be performed upon normal termination
    def __close_connection(self):
        traci.close()               #stop TraCI
        sys.stdout.flush()          #clear standard output

    def get_state_actions(self, state):
        return self._env[state].keys()

    def reset_episode(self):

        super(SUMO, self).reset_episode()

        #initialise TraCI
        traci.start([self._sumo_binary , "-c", self.__cfg_file, "-v", "true", "--random", "true"])

        #------------------------------------
        for vehID in self.get_vehicles_ID_list():
            self.__vehicles[vehID]['current_link'] = None
            self.__vehicles[vehID]['previous_node'] = self.__vehicles[vehID]['origin']
            self.__vehicles[vehID]['departure_time'] = -1.0
            self.__vehicles[vehID]['arrival_time'] = -1.0
            self.__vehicles[vehID]['travel_time'] = -1.0
            self.__vehicles[vehID]['time_last_link'] = -1.0
            self.__vehicles[vehID]['route'] = [self.__vehicles[vehID]['origin']]
            self.__vehicles[vehID]['initialized'] = False
            self.__vehicles[vehID]['n_of_traversed_links'] = 0

    def run_episode(self, max_steps=-1, mv_avg_gap=100):
        self.max_steps = max_steps
        self.start_time = datetime.now()
        print(f"Starting time: {self.start_time.strftime('%H:%M')}")
        self._has_episode_ended = False
        self.reset_episode()
        self._episodes += 1
        self.travel_times = np.array([])
        travel_avg_df = pd.DataFrame({"Step":[], "Average travel time":[]})
        cars_over_5k = occupation = occupation_df = occ_mea_init = occ_mea_end = occ_mea_int = self.__occ_dict = None
        higher_count = 0
        total_count = 0
        log_path = f"{os.getcwd()}/log/sim_{self.max_steps}_steps_{self.start_time.strftime('%d-%m-%y_%H-%M')}"
        over_5k = f"{log_path}/over_5k"
        self.sample_path = f"{log_path}/sample"
        teleport = f"{log_path}/teleports.txt"
        self.trips_per_od = {od : 0 for od in self.__od_pair_set}
        with_rl = self.__time_before_learning < max_steps
        
        if self.__flags['debug'] and self.__flags['plot_over5k']:
            cars_over_5k = pd.DataFrame({"Step":[], "Number of arrived cars over 5k":[]})
            occupation = {"Step":[]}
            occupation.update({edge.getID():[] for edge in self.__net.getEdges()})
            occupation_df = pd.DataFrame(occupation) 
            occ_mea_init = 5000
            occ_mea_end = 40000
            occ_mea_int = 100
            self.__occ_dict = {edge.getID(): list() for edge in self.__net.getEdges(withInternal=False)}

        if (self.__flags['over5k_log'] 
            or self.__flags['sample_log'] 
            or self.__flags['teleport_log']): self.__make_log_folder(log_path)
        if self.__flags['over5k_log']: self.__make_log_folder(over_5k)
        if self.__flags['sample_log']: self.__make_log_folder(self.sample_path)

        for vehID in self.get_vehicles_ID_list():
            routeID = 'r_' + vehID
            routeSet = set()
            if routeID not in routeSet:
                _, action = self._agents[vehID].take_action()
                traci.route.add(routeID, [action])
                routeSet.add(routeID)

        traci.simulation.subscribe((tc.VAR_TIME, tc.VAR_ARRIVED_VEHICLES_IDS, tc.VAR_DEPARTED_VEHICLES_IDS))
        self.current_time = traci.simulation.getTime()
        while self.current_time < self.max_steps:
            self.__vehicles_to_process_feedback = {}
            self.__vehicles_to_process_act = {}
            higher_per_step = 0

            if self.__flags['teleport_log'] : self.__update_teleport_log(teleport)
            self.sub_results = traci.simulation.getSubscriptionResults()

            [total_count, higher_per_step] = self.__process_arrived(total_count, higher_per_step)
            if (higher_per_step > 0 and self.__flags['plot_over5k'] and self.__flags['debug']):
                df = pd.DataFrame({"Step": [self.current_time],
                                   "Number of arrived cars over 5k": [higher_per_step]})
                cars_over_5k = cars_over_5k.append(df, ignore_index=True)

            self.__update_loaded_info()
            # departed vehicles (those that are entering the network)
            self.__update_departed_info()
            self.__process_all(over_5k)
            self.__process_vehicles_feedback(self.__vehicles_to_process_feedback)
            self.__process_vehicles_act(self.__vehicles_to_process_act, self.current_time)


            if (self.current_time == self.__time_before_learning 
                and self.__flags['sample_log'] 
                and self.__flags['debug']):
                self.__log_sample = self.__sample_log(self.sample_path)

            # if self.current_time > (self.max_steps / 2) and not_switched:
            #     for vehID in traci.vehicle.getIDList():
            #         self._agents[vehID].switch_epsilon(0)
            #     not_switched = False

            step = self.current_time
            if step % mv_avg_gap == 0 and step > 0 and (step >= self.__time_before_learning or not with_rl):
                df = pd.DataFrame({"Step": [step],
                                   "Average travel time": [self.travel_times.mean()]})
                travel_avg_df = travel_avg_df.append(df, ignore_index=True)
                self.travel_times = np.array([])
            if self.__flags['debug']:
                if step >= occ_mea_init and step <= occ_mea_end:
                    self.__measure_occupation()
                    if step % occ_mea_int == 0:
                        occupation = {"Step":[step]}
                        occupation.update(self.__get_edges_ocuppation())
                        occupation_df = occupation_df.append(pd.DataFrame(occupation), ignore_index=True)
                

            higher_count += higher_per_step
            traci.simulationStep()
            self.current_time = self.sub_results[tc.VAR_TIME]

        self.__close_connection()
        self.__write_sim_logfile(self.max_steps, total_count, higher_count)
        
        if self.__flags['debug']:
            sort = {key : self.trips_per_od[key] for key in sorted(self.trips_per_od)}
            frame = {"OD Pair": list(sort.keys()), "Number of Trips Ended": list(sort.values())}
            trips_dataframe = pd.DataFrame(frame)
            class_df = self.__create_class_dataframe()
        
        if self.__flags['plot']:
            travel_avg_df.plot(kind="scatter", x="Step", y="Average travel time")
            plt.xlabel("Step")
            plt.ylabel("Average travel time")

            if self.__flags['debug']: 
                plt.figure(1)
                trips_dataframe.plot(x="OD Pair", y="Number of Trips Ended", figsize=(15, 7), kind="bar")
                plt.subplots_adjust(left=0.05, bottom=0.20, right=0.95, top=0.95)
                plt.xlabel("OD Pair")
                plt.ylabel("Number of Trips Ended")
                
                plt.figure(2)
                class_df.plot(x="Interval", y="Trips Ended Within the Interval",kind='bar', figsize=(15, 7))
                plt.subplots_adjust(left=0.05, bottom=0.20, right=0.95, top=0.95)
                plt.xlabel("Interval")
                plt.ylabel("Trips Ended Within the Interval")

            plt.show()

        self.__save_to_csv("MovingAverage", travel_avg_df, with_rl)
        
        if self.__flags['debug']:
            self.__save_to_csv("ClassDivision", class_df, with_rl)
            self.__save_to_csv("TripsPerOD", trips_dataframe, with_rl)
            self.__save_to_csv("Occupation", occupation_df, with_rl)

            if self.__flags['plot_over5k']:
                cars_over_5k.plot(kind="scatter", x="Step", y="Number of arrived cars over 5k")
                plt.show()
                time_str = self.start_time.strftime('%d-%m-%y_%H-%M')
                plot_name = f"csv/CarsOver5k/sim_{self.max_steps}_steps_{time_str}.csv"
                cars_over_5k.to_csv(plot_name, index=False)

    def __process_vehicles_feedback(self, vehicles):
        # feedback_last
        for vehID in vehicles.keys():
            self._agents[vehID].process_feedback(vehicles[vehID][0], vehicles[vehID][1], vehicles[vehID][2], vehicles[vehID][3])
            self._agents[vehID].process_feedback(vehicles[vehID][0], vehicles[vehID][1], vehicles[vehID][2], vehicles[vehID][3], 1)

    def __process_vehicles_act(self, vehicles, current_time):

        # act_last
        for vehID in vehicles.keys():
            self.__get_infras_data(vehID, vehicles[vehID][0])

            use_C2I = 1

            _, action = self._agents[vehID].take_action(vehicles[vehID][0], vehicles[vehID][1], 1)

            if not vehicles[vehID][1]:
                self.__vehicles[vehID]["arrival_time"] = current_time
                self.__vehicles[vehID]["travel_time"] = self.__vehicles[vehID]["arrival_time"] - self.__vehicles[vehID]["departure_time"]
                continue

            # use only current link and action in updated route to avoid making
            # agent try to go back to the beggining of the route
            cur_route = [self.__vehicles[vehID]['route'][-2]+self.__vehicles[vehID]['route'][-1], action]

            traci.vehicle.setRoute(vehID, cur_route)

    def __is_link(self, edge_id):
        try:
            _ = self.__net.getEdge(edge_id)
            return True
        except:
            return False

    def has_episode_ended(self):
        return self._has_episode_ended

    def get_starting_edge_value(self, edge_id):
        # origin = self.__net.getNode(self.__get_edge_origin(edge_id))
        # dest = self.__net.getNode(self.__get_edge_destination(edge_id))
        # dist = np.linalg.norm(np.array(dest.getCoord()) - np.array(origin.getCoord()))
        # speed = self.__net.getEdge(edge_id).getSpeed()
        # base_value = dist / speed
        # return - base_value + rd.uniform(0, - base_value)
        return 0

    def __check_min_load(self, vehID):
        od_pair = self.__vehicles[vehID]['origin'] + self.__vehicles[vehID]['destination']
        self.__od_pair_load[od_pair] -= 1
        if self.__od_pair_load[od_pair] < self.__od_pair_min[od_pair]:
            routeID = f"r_{vehID}"
            if routeID not in traci.route.getIDList():
                self._agents[vehID].new_episode(self._agents[vehID].get_episode() + 1)
                _, action = self._agents[vehID].take_action()
                traci.route.add(routeID, [action])
            traci.vehicle.add(vehID, routeID)

    def __update_loaded_info(self):
        for vehID in traci.simulation.getLoadedIDList():
                self.__vehicles[vehID]['current_link'] = None
                self.__vehicles[vehID]['previous_node'] = self.__vehicles[vehID]['origin']
                self.__vehicles[vehID]['departure_time'] = -1.0
                self.__vehicles[vehID]['arrival_time'] = -1.0
                self.__vehicles[vehID]['travel_time'] = -1.0
                self.__vehicles[vehID]['time_last_link'] = -1.0
                self.__vehicles[vehID]['route'] = [self.__vehicles[vehID]['origin']]
                self.__vehicles[vehID]['initialized'] = False
                self.__vehicles[vehID]['n_of_traversed_links'] = 0
                od_pair = self.__vehicles[vehID]['origin'] + self.__vehicles[vehID]['destination']
                self.__od_pair_load[od_pair] += 1
                routeID = f"r_{vehID}"
                traci.vehicle.setRouteID(vehID, routeID)

    def __update_departed_info(self):
        for vehID in self.sub_results[tc.VAR_DEPARTED_VEHICLES_IDS]:
                self.__vehicles[vehID]["departure_time"] = self.current_time

    def __process_arrived(self, total_count, higher_per_step):
        for vehID in self.sub_results[tc.VAR_ARRIVED_VEHICLES_IDS]:
                self.__check_min_load(vehID)

                self.__vehicles[vehID]["arrival_time"] = self.current_time
                self.__vehicles[vehID]["travel_time"] = self.__vehicles[vehID]["arrival_time"] - self.__vehicles[vehID]["departure_time"]

                if self.__vehicles[vehID]["travel_time"] < 5000:
                    self.travel_times = np.append(self.travel_times, [self.__vehicles[vehID]["travel_time"]])

                if self.__vehicles[vehID]["travel_time"] >= 5000 : higher_per_step += 1

                reward = self.current_time - self.__vehicles[vehID]['time_last_link']
                reward *= -1
                if self.current_time > self.__time_before_learning:
                    self.__update_queue(self.__vehicles[vehID]['current_link'], reward)

                if (self.__vehicles[vehID]["route"][-1] == self.__vehicles[vehID]["destination"] 
                    and (self.current_time > self.__time_before_learning 
                         or self.__time_before_learning >= self.max_steps)):
                    total_count += 1
                    od_pair = self.__vehicles[vehID]["origin"] + self.__vehicles[vehID]["destination"] 
                    self.trips_per_od[od_pair] += 1
                    reward += 1000
                    if self.__flags['debug']:
                        index = math.floor(self.__vehicles[vehID]["travel_time"] / self.__class_interval)
                        if index >= len(self.__classifier):
                            self.__classifier[-1] += 1
                        else:
                            self.__classifier[index] += 1

                if self.current_time > self.__time_before_learning:
                    self.__vehicles_to_process_feedback[vehID] = [
                        reward,
                        self.__get_edge_destination(self.__vehicles[vehID]['current_link']),
                        self.__get_edge_origin(self.__vehicles[vehID]['current_link']),
                        self.__vehicles[vehID]['current_link']
                    ]
                if self.__flags['debug']:
                    if vehID in self.__log_sample and self.current_time > self.__time_before_learning:
                        od_pair = self.__vehicles[vehID]['origin'] + self.__vehicles[vehID]['destination']
                        path = self.sample_path + "/" + str(od_pair)
                        self.__write_veh_log(path, vehID, reward, True)
        
        return [total_count, higher_per_step]

    def __process_all(self, over5k_path):
        for vehID in self.get_vehicles_ID_list(): # all vehicles
                # who have departed but not yet arrived
                if self.__vehicles[vehID]["departure_time"] != -1.0 and self.__vehicles[vehID]["arrival_time"] == -1.0:
                    road = traci.vehicle.getRoadID(vehID)
                    if road != self.__vehicles[vehID]["current_link"] and self.__is_link(road): #but have just leaved a node
                        #update info of previous link
                        if self.__vehicles[vehID]['time_last_link'] > -1.0:
                            reward = self.current_time - self.__vehicles[vehID]['time_last_link']
                            reward *= -1

                            if self.current_time > self.__time_before_learning:
                                self.__update_queue(self.__vehicles[vehID]['current_link'], reward)
                                self.__vehicles_to_process_feedback[vehID] = [
                                    reward,
                                    self.__get_edge_destination(self.__vehicles[vehID]['current_link']),
                                    self.__get_edge_origin(self.__vehicles[vehID]['current_link']),
                                    self.__vehicles[vehID]['current_link']
                                ]

                            if self.__vehicles[vehID]['travel_time'] > 5000 and self.__flags['over5k_log']:
                                self.__write_veh_log(over5k_path, vehID, reward)

                            if self.current_time > self.__time_before_learning and self.__flags['debug']:
                                if vehID in self.__log_sample:
                                    od_pair = self.__vehicles[vehID]['origin'] + self.__vehicles[vehID]['destination']
                                    path = self.sample_path + "/" + str(od_pair)
                                    self.__write_veh_log(path, vehID, reward)


                        self.__vehicles[vehID]['time_last_link'] = self.current_time
                        self.__vehicles[vehID]['travel_time'] = self.current_time - self.__vehicles[vehID]['departure_time']

                        #update current_link
                        self.__vehicles[vehID]['current_link'] = road
                        self.__vehicles[vehID]['n_of_traversed_links'] += 1

                        #get the next node, and add it to the route
                        node = self.__get_edge_destination(self.__vehicles[vehID]["current_link"])
                        self.__vehicles[vehID]['route'].append(self.__get_edge_destination(self.__vehicles[vehID]['current_link']))

                        if node != self.__vehicles[vehID]['destination']:
                            if self.current_time > self.__time_before_learning:
                                outgoing = self.__net.getEdge(self.__vehicles[vehID]['current_link']).getOutgoing()
                                possible_actions = list()
                                for edge in outgoing:
                                    if ((len(edge.getOutgoing()) > 0 
                                        or self.__get_edge_destination(edge.getID()) == self.__vehicles[vehID]['destination'])):
                                        # and self.__get_edge_destination(edge.getID()) not in self.__vehicles[vehID]['route']
                                        possible_actions.append(edge.getID())
                                self.__vehicles_to_process_act[vehID] = [
                                    node, #next state
                                    possible_actions #available actions
                                ]
                            else:
                                self.__vehicles_to_process_act[vehID] = [
                                    node, #next state
                                    [self.__vehicles[vehID]['original_route'][len(self.__vehicles[vehID]['route']) - 1]] #available actions
                                ]

    def __update_queue(self, link_id, travel_time):
        comm_succ = rd.random()
        if comm_succ <= self.__comm_succ_rate:
            self.__comm_dev[link_id].append(travel_time)
            if len(self.__comm_dev[link_id]) > self.__max_queue:
                self.__comm_dev[link_id].pop(0)
    
    def __get_infras_data(self, vehID, node):
        comm_succ = rd.random()
        if comm_succ <= self.__comm_succ_rate:
            state = self.__net.getNode(node)
            for edge in state.getOutgoing():
                edge_id = edge.getID()
                destination = self.__get_edge_destination(edge_id)
                if self.__vehicles[vehID]['destination'] != destination:
                    ff_travel_time = edge.getLength() / edge.getSpeed()
                    possible_reward = traci.edge.getTraveltime(edge_id) - ff_travel_time
                    possible_reward = 0 if possible_reward < 0 else possible_reward
                    origin = self.__get_edge_origin(edge_id)
                    self._agents[vehID].process_feedback(possible_reward, destination, origin, edge_id, 1)
                # if len(self.__comm_dev[edge_id]) > 0 and self.__vehicles[vehID]['destination'] != destination:
                #     possible_reward = np.array(self.__comm_dev[edge.getID()]).mean()
                #     origin = self.__get_edge_origin(edge_id)
                #     self._agents[vehID].process_feedback(possible_reward, destination, origin, edge_id, 1)

    def __make_log_folder(self, folder_name):
        try:
            os.mkdir(folder_name)
        except OSError as e:
            if e.errno != 17:              
                print (f"Creation of the directory {folder_name} failed. Error message: {e.strerror}")
                self.__close_connection()
                sys.exit(-1)
    
    def __update_teleport_log(self, path):
        for vehID in traci.simulation.getStartingTeleportIDList():
                try:
                    with open(path, 'a') as txt_file:
                        teleport_str = f"Vehicle {vehID} in link {traci.vehicle.getRoadID(vehID)} "
                        teleport_str += f"teleported in step {self.current_time}\n"
                        txt_file.write(teleport_str)
                        txt_file.close()
                except IOError:
                    print(f"Unable to open {path} file")

    def __write_veh_log(self, path, vehID, reward, trip_end=False):
        filename = path + '/' + vehID + '.txt'
        try:
            with open(filename, 'a') as txt_file:
                c2i = 1
                rand_key = ''
                str_list = [
                    f"time step {self.current_time}: ",
                    f"Current state is {self.__vehicles[vehID]['route'][-1]}, ",
                    f"took action {self.__vehicles[vehID]['current_link']}, ",
                    f"with a reward of {reward}  ",
                ]
                QTable = self._agents[vehID].get_Q_table()
                if trip_end:
                    str_list.append(f"\nTrip ended with travel time {self.__vehicles[vehID]['travel_time']}\n\n")
                else:
                    max_val = max(QTable[c2i][self.__vehicles[vehID]['route'][-2]].values())
                    for key, val in QTable[c2i][self.__vehicles[vehID]['route'][-2]].items():
                        if val == max_val:
                            rand_key = key

                    if max_val == 0:
                        rand_key = self.__vehicles[vehID]['current_link']
                    str_list.append(f"\nNormal: {QTable[0][self.__vehicles[vehID]['route'][-1]]}  ")
                    str_list.append(f"\n   C2I: {QTable[1][self.__vehicles[vehID]['route'][-1]]}\n")
                str_list.insert(0,'' if rand_key == self.__vehicles[vehID]['current_link'] else '*')
                txt_file.write("".join(str_list))
                txt_file.close()
        except IOError:
            print(f"Couldn't open file {filename}")

    def __write_sim_logfile(self, total_steps, total_count, higher_count):
        try:
            with open('log/sims_log.txt', 'a') as logfile:
                end_time = datetime.now()
                str_list = [ 
                    "-----------------------------------------------\n",
                    f"Simulation with {total_steps} steps run in {self.start_time.strftime('%d/%m/%y')}\n",
                    "Communication success rate: ",
                    f"{self.__comm_succ_rate}"
                    "\n",
                    f"Start time: {self.start_time.strftime('%H:%M')}\n",
                    f"End time: {end_time.strftime('%H:%M')}\n",
                    f"Total trips ended: {total_count}\n",
                    f"Trips that ended with more than 5k steps: {higher_count}\n",
                    "Percentage (higher / total): {:.2f} %\n\n".format(higher_count / total_count * 100)
                ]
                logfile.write("".join(str_list))
                logfile.close()
        except IOError:
            print("Unable to open simulations log file")

    def __sample_log(self, sample_path):
        all_veh = traci.vehicle.getIDList()
        od_sample = {od:list() for od in self.__od_pair_set}
        sample = list()
        for veh in all_veh:
            od_pair = self.__vehicles[veh]['origin'] + self.__vehicles[veh]['destination']
            od_sample[od_pair].append(veh)

        for od_pair in od_sample.keys():
            sample.extend(rd.sample(od_sample[od_pair], 5))
            od_path = f"{sample_path}/{od_pair}"
            self.__make_log_folder(od_path)
        
        return sample

    def __create_data_classifier(self, interval, max):
        return list(map(lambda i:0, range(max // interval + 1)))

    def __create_class_dataframe(self):
        begin = 0
        class_dataframe = pd.DataFrame({"Interval":[], "Trips Ended Within the Interval":[]})
        while begin < self.__top_class_value:
            index = begin // self.__class_interval
            interval_name = str(begin) + ' - '
            begin += self.__class_interval - 1
            interval_name += str(begin)
            begin += 1
            aux_df = pd.DataFrame({"Interval":[interval_name], "Trips Ended Within the Interval": [self.__classifier[index]]})
            class_dataframe = class_dataframe.append(aux_df, ignore_index=True)

        aux_df = pd.DataFrame({"Interval":[f"{self.__top_class_value} or more"], "Trips Ended Within the Interval": [self.__classifier[-1]]})
        class_dataframe = class_dataframe.append(aux_df, ignore_index=True)
            
        return class_dataframe

    def __save_to_csv(self, folder_name, df, learning, idx=False):
        date_folder = self.start_time.strftime("%m_%d_%y")
        learning_str = "learning" if learning else "not_learning"
        succ = int(self.__comm_succ_rate * 100)
        try:
            os.mkdir(f"csv/{folder_name}/{date_folder}")
        except OSError as e:
            if e.errno != 17:
                print(f"Couldn't create folder {date_folder}, error message: {e.strerror}")
                return 
        try:
            os.mkdir(f"csv/{folder_name}/{date_folder}/{learning_str}")
        except OSError as e:
            if e.errno != 17:
                print(f"Couldn't create folder {learning_str}, error message: {e.strerror}")
                return 
        try:
            os.mkdir(f"csv/{folder_name}/{date_folder}/{learning_str}/{succ}")
        except OSError as e:
            if e.errno != 17:
                print(f"Couldn't create folder {learning_str}/{succ}, error message: {e.strerror}")
                return 
        str_list = [
            f"csv/{folder_name}/{date_folder}/{learning_str}/{succ}/sim_{self.max_steps}_steps_{self.start_time.strftime('%H-%M')}.csv"
        ]
        df.to_csv("".join(str_list), index=idx)

    def __measure_occupation(self):
        pass
        # for edge in self.__net.getEdges(withInternal=False):
        #     edge_ID = edge.getID()
            # self.__occ_dict[edge_ID].append(traci.edge.getLastStepOccupancy(edge_ID))

    def __get_edges_ocuppation(self):
        occ_avg = dict()
        for edge in self.__net.getEdges(withInternal=False):
            edge_ID = edge.getID()
            # occ_avg[edge_ID] = np.array(self.__occ_dict[edge_ID]).mean() 
            # self.__occ_dict[edge_ID].clear()
            occ_avg[edge_ID] = traci.edge.getLastStepVehicleNumber(edge_ID)

        return occ_avg

    def update_c2i_params(self, c2i_on = True, comm_succ_rate = 1):
        self.__comm_succ_rate = comm_succ_rate

