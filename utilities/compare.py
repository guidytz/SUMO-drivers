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
import matplotlib.pyplot as plt

if __name__ == "__main__":
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

    df = pd.read_csv(options.csv_file_1)
    col = list(df.columns)
    print(col)
    df = df.rename(columns={col[1]:"Without Learning"})
    df_aux = pd.read_csv(options.csv_file_2)
    df_aux = df_aux.rename(columns={col[1]:"With Learning"})
    df = df.join(df_aux, lsuffix='', rsuffix='_other')
    df = df.drop(columns=[col[0]+"_other"])
    # print(df.head())
    df.plot(kind="bar", x=col[0], y=["Without Learning", "With Learning"], figsize=(15, 8))
    plt.show()