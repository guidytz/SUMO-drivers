# prints headers of csv and their respective number

import argparse as ap
import csv
import sys

parser = ap.ArgumentParser()
parser.add_argument("-f", "--file", 
                    help="Path or name of the input csv")
args = parser.parse_args()

nome_arquivo = args.file

if args.file == None:
    sys.exit("Error! No file passed")

with open(nome_arquivo) as arquivo:
    colunas = csv.reader(arquivo)
    colunas = next(colunas) # reads the first line of the input csv, containing the column headers

i = 1
print(f"Column headers from {nome_arquivo} and their respective number:")
for coluna in colunas:
    print(f"{i} -> {coluna}")
    i += 1
