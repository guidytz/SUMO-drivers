import os
import sys
import argparse
import pandas as pd
import numpy
import matplotlib
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description='Script to take several csv files of same sort and take the average of the values to store')

    parser.add_argument("-d", "--dir", action="store", dest="dir_path",
                        help="Path to directory containing folders of simulations to compare (mandatory)")
    parser.add_argument("-s", "--std", action="store_true", dest="plot_std", default=False,
                        help="Flag to determine if standard deviation should be ploted")

    args = parser.parse_args()
    if not args.dir_path:
        print ('Wrong usage of script!')
        print ()
        parser.print_help()
        sys.exit()

    stream = os.popen(f"ls {args.dir_path}")
    output = stream.read()
    folders = output.split('\n')
    folders.pop()
    files = {folder:os.popen(f"ls {args.dir_path}/{folder}").read().split('\n') for folder in folders}
    df = dict()
    mean = dict()
    std = dict() if args.plot_std else None

    for folder in files.keys():
        df[folder] = pd.read_csv(f"{args.dir_path}/{folder}/{files[folder][0]}")["Step"].copy().to_frame()
        for file, i in  zip(files[folder], range(len(files[folder]) - 1)):
            csv_df_1 = pd.read_csv(f"{args.dir_path}/{folder}/{file}")
            df[folder] = df[folder].join(csv_df_1.set_index("Step"), on="Step")
            df[folder] = df[folder].rename(columns={df[folder].columns[-1]:f"file_{i}"})
        df[folder] = df[folder].set_index("Step")
        mean[folder] = df[folder].mean(axis=1, numeric_only=True)
        if args.plot_std: std[folder] = df[folder].std(axis=1, numeric_only=True)

    plt.figure()
    for folder in files.keys():
        plt.plot(mean[folder].index, mean[folder])
        if args.plot_std: plt.fill_between(
                                           std[folder].index, 
                                           mean[folder] - 2 * std[folder], 
                                           mean[folder] + 2 * std[folder], 
                                           alpha=0.5
                                          )
    
    plt.xlabel("Step")
    plt.ylabel("Average Travel Time")
    plt.legend(files.keys())
    plt.show()

if __name__ == "__main__":
    main()