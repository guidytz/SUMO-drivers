import argparse as ap
import csv
import sys
import random

parser = ap.ArgumentParser()
parser.add_argument("-f", "--arquivo", help="Nome do arquivo csv contendo os dados dos quais se quer gerar um csv menor. Tipo: string. Se o diretório do arquivo não for o mesmo do main.py, indicar o caminho para o arquivo. Exemplo no mesmo diretório do script: -f meuarquivo.csv. Exemplo em um diretório diferente: -f /home/user/caminho_para_arquivo/meuarquivo.csv")
parser.add_argument("-n", "--nlinhas", type=float, help="Número de linhas que o csv menor vai ter. Este número pode ser uma porcentagem a ser extraída do csv maior ou o número de linhas direto que vai ser extraído. Se for a porcentagem, o programa usará apenas a parte inteira do número de linhas que equivale à porcentagem indicada. Exemplo sem porcentagem: -n 10 (extrai 10 linhas). Exemplo com porcentagem: -n 10 (extrai 10%% das linhas do csv maior)") 
parser.add_argument("-p", "--percentage", action="store_true", default=False, help="Indica se o número passado pelo usuário é uma porcentagem")
args = parser.parse_args()

nome_arquivo = args.arquivo
num_linhas_csv = args.nlinhas
usa_porcentagem = args.percentage

if args.arquivo == None:
    sys.exit("Nenhum arquivo foi informado")

if args.nlinhas == None:
    sys.exit("Nenhum número foi informado")
elif num_linhas_csv <= 0:
    sys.exit("Número de linhas não pode ser negativo.")
elif usa_porcentagem and num_linhas_csv > 100:
    sys.exit("Porcentagem não pode ser superior a 100.")

with open(nome_arquivo) as arquivo:
    leitura = csv.reader(arquivo)

    linhas = []
    for linha in leitura:
        linhas.append(linha)

num_linhas_total = sum(1 for linha in linhas)

if usa_porcentagem:
    num_linhas_csv = int(num_linhas_total * (num_linhas_csv/100)) # tira a quantidade de linhas a ser extraídas pela porcentagem
else:
    num_linhas_csv = int(num_linhas_csv)

if num_linhas_csv > num_linhas_total:
    sys.exit("Número de linhas passado é maior que o do arquivo csv.")

n_linhas = random.sample(range(2, num_linhas_total+1), num_linhas_csv) # gera lista com números aleatórios de linhas do csv original

headers = linhas[0] # guarda primeira linha do arquivo csv original

linhas_csv = []
for numero in n_linhas:
    linhas_csv.append(linhas[numero]) # cria lista com as linhas que serão usadas no novo arquivo csv

nome_arquivo_csv = f"{nome_arquivo[:-4]}_com_{num_linhas_csv}_linhas.csv"

with open(nome_arquivo_csv, "w") as arqcsv: 
    escritacsv = csv.writer(arqcsv) # cria objeto de escrita do novo arquivo csv 
        
    escritacsv.writerow(headers) # escreve o cabeçalho
         
    escritacsv.writerows(linhas_csv) # escreve as linhas
        
print("Linhas usadas:")
for numero in n_linhas:
    print(numero)

print(f"Arquivo: {nome_arquivo_csv} gerado")

   
