'''
Created Date: Thursday, March 31th 2020, 15:10 am
Author: Guilherme Dytz dos Santos
-----
Last Modified: Wednesday, April 1st 2020, 12:24 pm
Modified By: guilhermedytz
'''
import sys
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    matplotlib.rcParams['figure.dpi'] = 50
    parser = argparse.ArgumentParser(
        description='Script to plot a bar graph using a csv file with two columns')

    parser.add_argument("-f1", "--file-1", action="store", dest="csv_file_1",
                        help="CSV file containing data of first file (mandatory)")
    parser.add_argument("-f2", "--file-2", action="store", dest="csv_file_2",
                        help="CSV file containing data of second file (mandatory)")

    options = parser.parse_args()
    if not options.csv_file_1 or not options.csv_file_2:
        print ('Wrong usage of script!')
        print ()
        parser.print_help()
        sys.exit()

    legend_1 = "SUMO DUA" if str(options.csv_file_1).find("not_learning") != -1 else "Q-Learning"
    legend_1 = f"{legend_1} with C2I" if str(options.csv_file_1).find("C2I") != -1 else f"{legend_1}"
    legend_2 = "SUMO DUA" if str(options.csv_file_2).find("not_learning") != -1 else "Q-Learning"
    legend_2 = f"{legend_2} with C2I" if str(options.csv_file_2).find("C2I") != -1 else f"{legend_2}"

    print(legend_1)
    print(legend_2)

    df = pd.read_csv(options.csv_file_1)
    col = list(df.columns)
    df = df.rename(columns={col[1]:legend_1})
    df_aux = pd.read_csv(options.csv_file_2)
    df_aux = df_aux.rename(columns={col[1]:legend_2})
    df = df.join(df_aux, lsuffix='', rsuffix='_other')
    df = df.drop(columns=[col[0]+"_other"])
    # print(df.head())
    df.plot(kind="bar", x=col[0], y=[legend_1, legend_2], figsize=(15, 7))
    plt.ylabel("Number of trips ended")
    plt.subplots_adjust(left=0.08, bottom=0.20, right=0.95, top=0.95)
    plt.show()