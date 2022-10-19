# generates smaller csv from input csv

import argparse as ap
import csv
import sys
import random

parser = ap.ArgumentParser()
parser.add_argument("-f", "--file", 
                    help="Path or name of input csv")
parser.add_argument("-n", "--nlines", type=float, 
                    help="Number of lines to be taken from the input csv or percentage of lines taken from input csv (depends on -p argument)") 
parser.add_argument("-p", "--percentage", action="store_true", default=False, 
                    help="Determines if the number passed by -n is a percentage. (default = false)")
args = parser.parse_args()

nome_arquivo = args.file
num_linhas_csv = args.nlines
usa_porcentagem = args.percentage

if args.file == None:
    sys.exit("Error!. No file passed")

if args.nlines == None:
    sys.exit("Error!. No number passed")
elif num_linhas_csv <= 0:
    sys.exit("Error!. Number cannot be negative")
elif usa_porcentagem and num_linhas_csv > 100:
    sys.exit("Error!. Percentage cannot be bigger than 100")

with open(nome_arquivo) as arquivo:
    leitura = csv.reader(arquivo)

    linhas = []
    for linha in leitura:
        linhas.append(linha)

num_linhas_total = sum(1 for linha in linhas)

if usa_porcentagem:
    num_linhas_csv = int(num_linhas_total * (num_linhas_csv/100)) # caluclates number of lines to be taken if number is percentage
else:
    num_linhas_csv = int(num_linhas_csv)

if num_linhas_csv > num_linhas_total:
    sys.exit("Error! Number of lines passed is bigger than the total number of lines from input csv")

n_linhas = random.sample(range(2, num_linhas_total+1), num_linhas_csv) # generates list with random line numbers

headers = linhas[0]

linhas_csv = []
for numero in n_linhas:
    linhas_csv.append(linhas[numero])

nome_arquivo_csv = f"{nome_arquivo[:-4]}_{num_linhas_csv}_lines.csv"

with open(nome_arquivo_csv, "w") as arqcsv: 
    escritacsv = csv.writer(arqcsv)
    escritacsv.writerow(headers) 
    escritacsv.writerows(linhas_csv) 
        
print("Linhas used:")
for numero in n_linhas:
    print(numero)

print(f"File: {nome_arquivo_csv} generated")

   
