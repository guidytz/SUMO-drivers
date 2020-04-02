'''
Created Date: Thursday, March 26th 2020, 10:11 am
Author: Guilherme Dytz dos Santos
-----
Last Modified: Thursday, March 26th 2020, 11:37 am
Modified By: guilhermedytz
'''
import os
import sys
import argparse
import numpy as np
import re

def print_results(folder_path):
    ls_command = "ls " + folder_path + "sample"
    stream = os.popen(ls_command)
    output = stream.read()
    od_folders = output.split('\n')
    od_folders.pop()

    for folder in od_folders:
        ls_inside = ls_command + "/" + folder
        stream = os.popen(ls_inside)
        output = stream.read()
        files = output.split('\n')
        files.pop()

        print(folder + ':', end=' ')
        n_trips_list = list()
        sp = re.split('\d+', folder)
        sp.pop()
        number_pos = folder.find(sp[-1]) + len(sp[-1])
        end_node = sp[-1] + folder[number_pos]
        for txt in files:
            path = folder_path + 'sample/' + folder + "/" + txt
            try:
                with open(path, 'r') as txt_file:
                    string = txt_file.read()
                    trips = string.split('\n\n')
                    wrong_end = 0
                    for trip in trips:
                        lines = trip.split('\n')
                        if lines[-1].find("ended") != -1:
                            if lines[-2].find(end_node) == -1:
                                wrong_end += 1
                        else:
                            wrong_end += 1
                    n_trips = len(trips) - wrong_end
                    print(n_trips, end=' ')
                    n_trips_list.append(n_trips)
            except IOError:
                print("Unable to open", txt_file)
            
        n_trips_array = np.array(n_trips_list)
        print("Avg:", n_trips_array.mean(), end=' ')
        print() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to count number of ended trips per OD pair')

    parser.add_argument("-p", "--path", action="store", dest="log_path",
                        help="Path to log files containing OD pair folders")
    parser.add_argument("-a", "--all", action="store_true", dest="is_all",
                        default=False, help="Count for all simulation logs")

    options = parser.parse_args()
    if options.is_all:
        stream = os.popen("ls -d log/*/")
        output = stream.read()
        sim_folders = output.split('\n')
        sim_folders.pop()
        for folder in sim_folders:
            print(folder)
            print_results(folder)
            print()
    elif not options.log_path:
        print ('Wrong usage of script!')
        print ()
        parser.print_help()
        sys.exit()
    else:
        print_results(options.log_path)