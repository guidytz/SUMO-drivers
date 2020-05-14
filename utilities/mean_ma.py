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

    parser.add_argument("-f", "--folder", action="store", dest="folder_path",
                        help="Path to folder containing csv files to take average (mandatory)")

    args = parser.parse_args()
    if not args.folder_path:
        print ('Wrong usage of script!')
        print ()
        parser.print_help()
        sys.exit()

    stream = os.popen(f"ls {args.folder_path}")
    output = stream.read()
    files = output.split('\n')
    files.pop()

    df = pd.read_csv(f"{args.folder_path}/{files[0]}")["Step"].copy().to_frame()
    for file, i in  zip(files, range(len(files))):
        csv_df = pd.read_csv(f"{args.folder_path}/{file}")
        df = df.join(csv_df.set_index("Step"), on="Step")
        df = df.rename(columns={df.columns[-1]:f"file_{i}"})

    df = df.set_index("Step")
    mean = df.mean(axis=1, numeric_only=True)
    # mean = mean.rename(columns={mean.columns[-1]:f"Mean Values"})
    std = df.std(axis=1, numeric_only=True)
    # std = std.rename(columns={std.columns[-1]:f"Standard Deviation"})
    # full_df = mean.join(std, on="Step").reset_index()

    # columns = list(mea.columns)
    plt.figure()
    plt.plot(mean.index, mean)
    plt.fill_between(std.index, mean - 2 * std, mean + 2 * std, alpha=0.2)
    # full_df.plot(kind="line", x=columns[0], y=columns[1])
    plt.show()



if __name__ == "__main__":
    main()