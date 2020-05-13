import os
import sys
import argparse
import pandas as pd
import numpy
import matplotlib
import matplotlib.pyplot as plt


def main():
    matplotlib.rcParams['figure.dpi'] = 80
    parser = argparse.ArgumentParser(
        description='Script to take several csv files of same sort and take the average of the values to store')

    parser.add_argument("-f1", "--folder-1", action="store", dest="folder_path_1",
                        help="Path to folder containing csv files to take average (mandatory)")
    parser.add_argument("-f2", "--folder-2", action="store", dest="folder_path_2",
                        help="Path to folder containing csv files to take average (mandatory)")

    args = parser.parse_args()
    if not args.folder_path_1 and not args.folder_path_2:
        print ('Wrong usage of script!')
        print ()
        parser.print_help()
        sys.exit()

    stream = os.popen(f"ls {args.folder_path_1}")
    output = stream.read()
    files_1 = output.split('\n')
    files_1.pop()

    stream = os.popen(f"ls {args.folder_path_2}")
    output = stream.read()
    files_2 = output.split('\n')
    files_2.pop()

    df_1 = pd.read_csv(f"{args.folder_path_1}/{files_1[0]}")["Step"].copy().to_frame()
    for file, i in  zip(files_1, range(len(files_1))):
        csv_df_1 = pd.read_csv(f"{args.folder_path_1}/{file}")
        df_1 = df_1.join(csv_df_1.set_index("Step"), on="Step")
        df_1 = df_1.rename(columns={df_1.columns[-1]:f"file_{i}"})

    df_2 = pd.read_csv(f"{args.folder_path_2}/{files_2[0]}")["Step"].copy().to_frame()
    for file, i in  zip(files_2, range(len(files_2))):
        csv_df_2 = pd.read_csv(f"{args.folder_path_2}/{file}")
        df_2 = df_2.join(csv_df_2.set_index("Step"), on="Step")
        df_2 = df_2.rename(columns={df_2.columns[-1]:f"file_{i}"})

    df_1 = df_1.set_index("Step")
    mean_1 = df_1.mean(axis=1, numeric_only=True)
    std_1 = df_1.std(axis=1, numeric_only=True)
    df_2 = df_2.set_index("Step")
    mean_2 = df_2.mean(axis=1, numeric_only=True)
    std_2 = df_2.std(axis=1, numeric_only=True)


    plt.figure()
    plt.plot(mean_1.index, mean_1)
    plt.fill_between(std_1.index, mean_1 - 2 * std_1, mean_1 + 2 * std_1, alpha=0.2)
    plt.plot(mean_2.index, mean_2)
    plt.fill_between(std_2.index, mean_2 - 2 * std_2, mean_2 + 2 * std_2, alpha=0.2)
    plt.legend(["Q-Learning", "Q-Learning with C2I"])
    plt.show()



if __name__ == "__main__":
    main()