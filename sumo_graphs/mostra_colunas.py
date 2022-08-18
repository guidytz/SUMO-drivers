import argparse as ap
import csv
import sys

parser = ap.ArgumentParser()
parser.add_argument("-f", "--arquivo", help="Nome do arquivo csv contendo os dados dos quais se quer saber as colunas. Tipo: string. Se o diretório do arquivo não for o mesmo do main.py, indicar o caminho para o arquivo. Exemplo no mesmo diretório do script: -f meuarquivo.csv. Exemplo em um diretório diferente: -f /home/user/caminho_para_arquivo/meuarquivo.csv")
args = parser.parse_args()

nome_arquivo = args.arquivo

if args.arquivo == None:
    sys.exit("Nenhum arquivo foi informado")

with open(nome_arquivo) as arquivo:
    colunas = csv.reader(arquivo)
    colunas = next(colunas) # lê a primeira linha do arquivo csv, que contém os colunas que podem ser usados

i = 1
print(f"Colunas de {nome_arquivo} e seu respectivo número, utilizado para indicar as colunas usadas nos atributos do grafo:")
for coluna in colunas:
    print(f"{i} -> {coluna}")
    i += 1
