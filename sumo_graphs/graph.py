# cria grafo com dados lidos de um csv

import igraph as ig
import sys
import os
import itertools
import string
import itertools
import sys
import datetime as dt
import matplotlib.pyplot as plt
import math
from csv import DictReader
from pandas import DataFrame
from collections import Counter
from decimal import Decimal, getcontext

# == Funções para as diferentes partes do programa == 

# - Importação de dados -

# le csv inteiro e o guarda na memória, após, salva cada linha da leitura em uma lista
# verifica se string é um númerico ou não 
def eh_numero(num_str: string) -> bool:
    try:
        float(num_str)
    except ValueError:
        return False
    return True

# recebe um dicionário e as keys deste dicionário, verificando se existe nele alguma entrada vazia
def tem_atributo_vazio(linha: dict, keys: list) -> bool:
    for key in keys:
        if linha[key] == "":
            return True
    return False

# lê os dados do csv e retorna cada linha como um dicionário cujas keys são a primeira coluna do csv e uma lista com as keys do dicionário
def importa_csv(caminho_csv):  # recebe o caminho e o nome do arquivo a ser lido
    with open(caminho_csv) as arquivo:
        # lê cada atributo como string, que será convertido posteriormente
        leitura = DictReader(arquivo)
        keys = leitura.fieldnames  # guardas as keys do dicionário

        linhas = []
        id = 0  # identificador usado para criar arestas
        num_linha = 2
        for linha in leitura:
            if not tem_atributo_vazio(linha, keys): # não inclui na lista de dicionários algum dicionário que tiver atributo vazio
                linhas.append(linha) # grava cada linha (um dicionário próprio) em uma lista de linhas
                linha["id"] = id
                id += 1
            else:
                print(f"Linha {num_linha} contém atributo vazio e não será considerada")
            num_linha += 1

        # converte atributos do dicionário para seus tipos respectivos (inicialmente são strings)
        for linha in linhas:
            for key in keys:
                if eh_numero(linha[key]):
                    linha[key] = float(linha[key]) # converte os atributos numéricos para float
    
    return linhas, keys  # retorna lista contendo dicionários e outra lista com as keys desses dicionários (são todas as mesmas)

# normaliza os valores da lista de dicionários (com a fórmula n = (x-min)/(max-min)) e retorna uma lista de dicionários normalizada
def normaliza_lista(lista_atributos):  # normaliza os valores de uma lista
    minimo = min(lista_atributos)
    maximo = max(lista_atributos)

    lista_atributos_norm = []

    if (maximo-minimo) == 0:
        exit("Erro na normalização: divisão por zero")

    for atributo in lista_atributos:
        lista_atributos_norm.append((atributo-minimo)/(maximo-minimo))

    return lista_atributos_norm

# normaliza os valores do dicionário inteiro
def normaliza_lista_dict(lista_dicionarios, keys, lista_ids_label):
    lista_dicionarios_norm = lista_dicionarios

    keys_norm = []
    for key in keys:
        if key not in lista_ids_label:  # só os atributos que não compõem o label serão normalizados
            lista_atributos = []  # lista de atributos, usada para calcular fórmula da normalização

            for dicionario in lista_dicionarios_norm:
                # guarda todos os valores de uma key do dicionário na lista
                lista_atributos.append(dicionario[key])

            lista_atributos = normaliza_lista(lista_atributos)

            for dicionario, atributo_norm in zip(lista_dicionarios_norm, lista_atributos):
                # para cada dicionário na lista, atribui o valor respectivo da lista de atributos normalizados
                dicionario[f"{key} Norm"] = atributo_norm

            # monta lista com nomes dos atributos normalizados
            keys_norm.append(key + " Norm")
        else:
            keys_norm.append(key)

    # retorna a lista de dicionários normalizada
    return lista_dicionarios_norm, keys_norm

#  - Processamento da entrada - 

# recebe a lista contendo os vértices do grafo e os nomes dos atributos que irão compor o id, retornando a lista com os nomes dos atributos concatenados
def cria_lista_ids(lista_dicionarios, atributos_usados):
    lista_ids = []

    for nodo in lista_dicionarios:
        nomes_atributos = []
        # para cada aributo, forma string que irá compor o nome do vértice
        for i in range(len(atributos_usados)):
            if i == 0:  # o primeiro atributo do nome não recebe "_" antes
                nomes_atributos.append(f"{nodo[atributos_usados[i]]}")
            else:
                nomes_atributos.append(f"_{nodo[atributos_usados[i]]}")
        nome_atributo = ""
        for nome in nomes_atributos:
            nome_atributo += nome  # monta o nome do vértice com as strings geradas por cada atributo
        # monta a lista contendo os nomes dos vértices
        lista_ids.append(nome_atributo)

    return lista_ids

# recebe a lista contendo os ids dos nodos e retorna True se todos forem diferentes ou False caso contrário
def ids_validos(lista_ids):
    pares_ids = itertools.combinations(lista_ids, 2)  # cria pares de ids

    for par in pares_ids:
        if par[0] == par[1]:  # se qualquer par for igual, retorna False
            return False

    return True  # caso contrário, retorna True

# recebe uma lista de atributos (e.g. a lista de atributos usados para unir vértices) e monta uma string com estes atributos separados por "-"
def cria_string_com_atributos(lista_atributos):
    string_atributos = ""
    for i in range(len(lista_atributos)):
        if i == len(lista_atributos)-1:
            string_atributos += str(lista_atributos[i])
        else:
            string_atributos += f"{str(lista_atributos[i])}-"
    return string_atributos

# recebe os parâmetros do usuário e gera o nome do arquivo que contém alguns dados do grafo
def monta_nome(grafo, limiar, lista_atributos):
    tempo = dt.datetime.now()
    hora_atual = tempo.strftime('%H%M%S')

    atributos = cria_string_com_atributos(lista_atributos)

    limiar_str = str(limiar)
    # remove o ponto do limiar para nao causar problemas com o nome e a extensão do arquivo
    limiar_str_processada = limiar_str.replace('.', "'")
    nome_final = f"{hora_atual}_atb{atributos}_l{limiar_str_processada}"

    return nome_final

# converte string representando intervalo numérico na forma "início-fim" em uma lista contendo os números naquele intervalo
def converte_intervalo(intervalo):
    numeros = intervalo.split("-")

    inicio = int(numeros[0])
    fim = int(numeros[1])

    if inicio > fim:
        sys.exit("Erro: início do intervalo maior do que o final")

    lista_intervalo = []

    x = inicio
    while x != fim+1:  # preenche a lista incrementando os números do início do intervalo ao fim
        lista_intervalo.append(x)
        x += 1

    return lista_intervalo

# determina se a entrada é um intervalo numérico na forma "início-fim" ou uma lista de inteiros, retornando a lista de inteiros que corresponde ao intervalo ou a própria lista de inteiros
def processa_int_ou_intervalo(entrada):
    lista_intervalo = []

    for v in entrada:
        if "-" in v:
            for num in converte_intervalo(v):
                lista_intervalo.append(num)
        else:
            lista_intervalo.append(int(v))  # transforma os números em inteiros

    return lista_intervalo

# - Criação de arestas -

# dependendo da lógica escolhida, verifica se deve ser criada uma aresta entre um par de nodos
def verifica_aresta(lista_resultado, usar_or):
    if usar_or:
        for resultado in lista_resultado:
            if resultado == 1:  # dada a lista final de resultados, se houver algum verdadeiro, a aresta é criada
                return True
        return False
    else:
        for resultado in lista_resultado:
            if resultado == 0:  # dada a lista final de resultados, se houver algum falso, a aresta não é criada
                return False
        return True

# dados dois vértices e uma lista de atributos usados como restrição, a função retorna -1 se a lista de restrições contiver "none",
# False se os vértices possuírem os mesmos valores para os mesmos atributos restritivos ou True se possuírem todos os valores diferentes
# para os mesmos atributos restritivos
def dentro_restricao(v1, v2, lista_restricoes):
    for restricao in lista_restricoes:
        if restricao == "none":
            return -1
        if v1[restricao] == v2[restricao]:
            return False
    return True

# verifica se um atributo entre dois dicionários está dentro do limiar ou não
# recebe dois dicionários, um atributo, um limiar e a precisão da diferença entre atributos
def dentro_limiar(v1: dict, v2: dict, atributo: string, limiar: float, precisao: float) -> bool:
    getcontext().prec = precisao 
    return abs(Decimal(v1[atributo]) - Decimal(v2[atributo])) <= limiar

# monta uma lista de arestas a partir de uma lista de atributos, uma de dicionários, uma de restrições, uma lógica para montar arestas e um limiar
def monta_arestas(atributos, lista_dicionarios, lista_restricoes, usar_or, limiar, precisao):
    arestas = []
    pesos_arestas = []
    # para cada par de dicionários da lista
    for v1, v2 in itertools.combinations(lista_dicionarios, 2):
        # indica se para cada atributo, deve haver uma aresta (1) ou não (0)
        lista_resultados = []
        # não é usada nenhuma restrição
        if dentro_restricao(v1, v2, lista_restricoes) == -1:
            for atributo in atributos:
                # se a diferença absoluta do valor do atributo de dois nodos for menor ou igual ao limiar
                if dentro_limiar(v1, v2, atributo, limiar, precisao):
                    # lista de resultados para aquele par contém 1, isto é, verdadeiro
                    lista_resultados.append(1)
                else:
                    lista_resultados.append(0)  # caso contrário contém zero

            if verifica_aresta(lista_resultados, usar_or):
                # adiciona a aresta à lista de arestas, utilizando o identificador numérico dos vértices
                arestas.append((v1["id"], v2["id"]))
                # como, para cada atributo, a lista contém 1 se há aresta e 0 se não há, a soma desses uns dará o número de atributos dentro do limiar entre o par de nodos
                pesos_arestas.append(sum(lista_resultados))

        else:  # considerando a restrição
            # se os vértices estiverem dentro da restrição
            if dentro_restricao(v1, v2, lista_restricoes):
                for atributo in atributos:
                    if dentro_limiar(v1, v2, atributo, limiar, precisao):
                        lista_resultados.append(1)
                    else:
                        lista_resultados.append(0)

                if verifica_aresta(lista_resultados, usar_or):
                    arestas.append((v1["id"], v2["id"]))
                    pesos_arestas.append(sum(lista_resultados))

    # retorna lista com arestas e lista com pesos das arestas
    return arestas, pesos_arestas

# - Toma medidas sobre o grafo - 

# determina se lista de medidas possui alguma medida considerada custosa
def determina_possui_medida_custosa(lista_medidas, medidas_custosas):
    for medida_custosa in medidas_custosas:
        if medida_custosa in lista_medidas:
            return True
    return False

# calcula medidas de centralidade do grafo, retornando um dicionário com as medidas
def calcula_medidas(grafo, lista_medidas, lista_ids):
    for medida in lista_medidas:
        lista_medida_calculada = getattr(grafo, medida)()

        if medida == lista_medidas[0]:
            lista_medida = lista_medida_calculada.copy()  # usada para ordenar tabela
            lista_medida.sort(reverse=True)

            lista_ids_num_ordenada = []
            for m_ord in lista_medida:
                for i in range(grafo.vcount()):
                    if m_ord == lista_medida_calculada[i] and i not in lista_ids_num_ordenada:
                        lista_ids_num_ordenada.append(i)

            lista_ids_ordenada = []
            for id in lista_ids_num_ordenada:  # traduz o id numérico para o id do label do grafo
                lista_ids_ordenada.append(lista_ids[id])

            # mostra as medidas em ordem decrescente de degree
            dict_valores_medidas = {"id": lista_ids_ordenada}

        lista_medida_calculada = list(
            map(lambda x: round(x, 4), lista_medida_calculada))

        lista_medida_calculada_ordenada = []
        for i in lista_ids_num_ordenada:
            lista_medida_calculada_ordenada.append(lista_medida_calculada[i])

        dict_valores_medidas[medida] = lista_medida_calculada_ordenada

    return dict_valores_medidas

# input: graph representing network composed of nodes that are dictionaries
# output: dictionary with the frequency that the specified key appears in the graph
def calculate_frequency_keys(graph, chosen_key):
    list_keys = []
    for v in graph.vs:
        list_keys.append(v[chosen_key])

    dict_freq_keys = dict(Counter(key for key in list_keys))

    sorted_dict_freq_keys = {item[0]: item[1] for item in sorted(dict_freq_keys.items(), key=lambda x: (-x[1], x[0]))}  # creates sorted dictionary by value in key in decreasing order

    processed_dict = dict()
    processed_dict[chosen_key] = sorted_dict_freq_keys.keys()

    frequency_list = []
    for item in list(sorted_dict_freq_keys.items()):
        frequency_list.append(item[1])

    processed_dict["frequency"] = frequency_list

    return processed_dict

# recebe os dados em uma lista, o nome que o arquivo de saída terá e se o dados se referem 
# às medidas de centralidade ("centrality") ou frequência ("frequency")
def monta_tabela(dados: list, nome: str, tipo: str) -> None:
    create_directory("results")
    create_directory("results/tables")
    local_nome = f"results/tables/{tipo}/{nome}"

    create_directory("results/tables/centrality") if tipo == "centrality" else create_directory("results/tables/frequency")

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.axis("tight")
    
    colunas = dados
    df = DataFrame(colunas, columns=colunas)

    ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="upper center")

    plt.savefig(f"{local_nome}", format="pdf", bbox_inches="tight")

# - Visual do grafo - 

# calcula o tamanho da imagem do grafo
def calcula_bbox(n_arestas):
    bbox_limit = 3000

    # criada com raiz quadrada, chegando perto dos pontos
    bbox = math.sqrt(n_arestas)*150

    if bbox > bbox_limit:  # define limite para bbox
        bbox = bbox_limit

    return (bbox, bbox)

# determina cor de um vértice, quão maior o degree, mais quente a cor
def determina_cor_vertice(grau, media):
    razao = grau / media

    if razao < 0.1:
        return "#ADD8E6"  # light blue
    elif razao < 0.3:
        return "#0000CD"  # medium blue
    elif razao < 0.5:
        return "#0000FF"  # blue
    elif razao < 0.7:
        return "#90EE90"  # light green
    elif razao < 0.9:
        return "#00FF00"  # green
    elif razao < 1.1:
        return "#006400"  # dark green
    elif razao < 1.3:
        return "#FFFF00"  # yellow
    elif razao < 1.5:
        return "#FFCC00"  # dark yellow
    elif razao < 1.7:
        return "#FFA500"  # orange
    elif razao < 1.9:
        return "#FF8C00"  # dark orange
    elif razao < 2.1:
        return "#FF4500"  # orange red
    elif razao < 2.3:
        return "#FF3333"  # light red
    elif razao < 2.5:
        return "#FF0000"  # red
    elif razao < 2.7:
        return "#8B0000"  # dark red
    elif razao < 2.9:
        return "#9370DB"  # medium purple
    elif razao < 3.1:
        return "#A020F0"  # purple
    else:
        return "#000000"  # black

# cria lista de cores dos vértices do grafo
def lista_cores(g):
    lista_graus = g.degree()

    if len(lista_graus) == 0:
        media_graus = 0
    else:
        media_graus = sum(lista_graus) / len(lista_graus)

    lista_cores = []
    for grau in lista_graus:
        lista_cores.append(determina_cor_vertice(grau, media_graus))

    return lista_cores

# determina características visuais do grafo
def determine_visual_style(g):
    font_lower_bound = 8

    visual_style = {}
    visual_style["bbox"] = calcula_bbox(g.ecount())
    visual_style["margin"] = 60
    visual_style["edge_color"] = "grey"
    visual_style["vertex_color"] = lista_cores(g)
    visual_style["vertex_label_dist"] = 1.1
    # tamanho do vértice será o maior número entre 15 e 6*math.sqrt(d)
    visual_style["vertex_size"] = [max(15, 6*math.sqrt(d)) for d in g.degree()]
    # tamanho da fonte entre 10 e 8, dependendo do grau do vértice
    visual_style["vertex_label_size"] = [max(10, 4*math.sqrt(d) if 4*math.sqrt(d) < font_lower_bound else font_lower_bound) for d in g.degree()]
    visual_style["layout"] = g.layout("auto")

    return visual_style

# determina se o grafo possui algum vértice com grau mínimo passado
def possui_grau_minimo(grafo, d_minimo):
    for v in grafo.vs:
        if v.degree() < d_minimo:
            return True
    return False

# input: name of directory to be created
# output: creates directory with the name passed
def create_directory(dirname: str) -> None:
    if not os.path.exists(dirname):
        try:
            os.mkdir(f"{dirname}")
        except OSError:
            raise OSError(f"Não foi possível criar o diretório {dirname}")

def calcula_max_step(lista_dict, keys):
    nome_step = list(filter(lambda x: x == "Step" or x == "step", keys))[0]
    
    lista_step = []
    for v in lista_dict:
        lista_step.append(v[nome_step])

    return int(max(lista_step))

def verifica_se_esta_no_intervalo(limite_inferior: int, limite_superior: int, valor: float, ultimo_intervalo: bool) -> bool:
    if not ultimo_intervalo:
        if valor >= limite_inferior and valor < limite_superior:
            return True
        else:
            return False
    else:
        if valor >= limite_inferior and valor <= limite_superior:
            return True
        else:
            return False

# recebe uma lista de vértices com mesmo link e retorna uma lista contendo todos os vizinhos desse link
def retorna_vizinhos_link(lista_vertices_link: list, link: str, nome_link: str) -> list:
    lista_vizinhos_link = []
    for v_link in lista_vertices_link:
        lista_vizinhos = v_link.neighbors()
        for vizinho_link in lista_vizinhos:
            if vizinho_link[nome_link] != link and vizinho_link not in lista_vizinhos_link:
                lista_vizinhos_link.append(vizinho_link)

    return lista_vizinhos_link

# recebe uma lista de vizinhos do link e filtra os links vizinhos que estão no intervalo passado
def retorna_vizinhos_no_intervalo(lista_vizinhos_link: list, intervalo: tuple, ultimo_intervalo: bool, nome_step: str, nome_link: str) -> list:
    limite_inferior = intervalo[0]
    limite_superior = intervalo[1]
    
    # para cada vertice, verificar se esta no intervalo e adicionar à lista
    lista_vizinhos_no_intervalo = []
    for vizinho in lista_vizinhos_link:
        if verifica_se_esta_no_intervalo(limite_inferior, limite_superior, vizinho[nome_step], ultimo_intervalo):
            #vizinho_tupla = (vizinho[nome_link], vizinho[nome_step])
            if vizinho[nome_link] not in lista_vizinhos_no_intervalo:
                lista_vizinhos_no_intervalo.append(vizinho[nome_link]) # quantas informações dos vizinhos guardar

    return lista_vizinhos_no_intervalo

# cria um dicionário contendo todos os links do grafo e seus vizinhos em determinado intervalo de tempo
def cria_dicionario_vizinhos_links(grafo: ig.Graph, keys: list, intervalo: int, max_step: int) -> dict:
    dict_vizinhos = dict()

    nome_link = list(filter(lambda x: x == "Link" or x == "link", keys))[0]
    nome_step = list(filter(lambda x: x == "Step" or x == "step", keys))[0]

    lista_links = []
    for v in grafo.vs:
        lista_links.append(v[nome_link])

    lista_links_unica = []
    for link in lista_links:
        if link not in lista_links_unica:
            lista_links_unica.append(link)

    lista_intervalos = [] # constroi lista de intervalos (compostos por uma tupla) utilizando o último step e o tamanho do intervalo selecionado: ["[0-1)", "[1-2) ... [n-1, n]"] 
    nome_intervalo = (0, 0)  # obs: intervalo é fechado à esquerda e aberto à direita, mas o último intervalo é fechado nos dois lados
    limite_inferior = 0
    limite_superior = 0
    if intervalo == 0:
        exit("Erro, intervalo não pode ser zero")
    num_intervalos = int(max_step / intervalo)
    for i in range(num_intervalos):
        limite_superior += intervalo
        nome_intervalo = (limite_inferior, limite_superior)
        limite_inferior += intervalo
        lista_intervalos.append(nome_intervalo)

    for link in lista_links_unica:
        dict_vizinhos[link] = dict()

        lista_link = list(filter(lambda v: v[nome_link] == link, grafo.vs)) # lista de vértices do grafo com link especificado
        lista_vizinhos_link = retorna_vizinhos_link(lista_link, link, nome_link)

        for i in range(len(lista_intervalos)):
            ultimo_intervalo = True if i == num_intervalos-1 else False # se for o último intervalo, ele será fechado dos dois lados
            dict_vizinhos[link][lista_intervalos[i]] = retorna_vizinhos_no_intervalo(lista_vizinhos_link, lista_intervalos[i], ultimo_intervalo, nome_step, nome_link)
    
    return dict_vizinhos

def generate_graph_neighbours_dict(nome_arquivo_csv: str, lista_atributos_numerico: list, lista_ids_label_numerico: list, lista_restricoes_numerico: list, 
                                   limiar: float, usar_or: bool, lista_medidas: list, nao_gerar_imagem_grafo: bool, usar_grafo_puro: bool, giant_component: bool,
                                   use_raw_data: bool, min_degree: int, min_step: int, arestas_para_custoso: int, precisao: int) -> dict:

    # == Processa listas numéricas ==

    if lista_atributos_numerico != ["ALL"]:
        lista_atributos_numerico = processa_int_ou_intervalo(lista_atributos_numerico)

    lista_ids_label_numerico = processa_int_ou_intervalo(lista_ids_label_numerico)

    if lista_restricoes_numerico != ["none"]:
        lista_restricoes_numerico = processa_int_ou_intervalo(lista_restricoes_numerico)

    # == Lê csv, traduz entrada numérica dos ids para atributos e normaliza dados, se foi pedido ==

    print("Lendo arquivo...")
    lista_dict, keys = importa_csv(nome_arquivo_csv)

    lista_ids_label = [] # usada como label do grafo, indica também atributos que não serão normalizados
    for num_id in lista_ids_label_numerico:  # traduz os número passados como argumento correspondente às colunas
        lista_ids_label.append(keys[num_id-1]) # numeração das colunas começa em 1, por isso -1

    if not use_raw_data:
        lista_dict, keys = normaliza_lista_dict(lista_dict, keys, lista_ids_label)
    print("Arquivo lido.")

    # Traduz entrada numérica dos outros parâmetros

    print("\nTraduzindo atributos...")

    atributos = []
    todos = False
    if lista_atributos_numerico != ["ALL"]:
        for num_atb in lista_atributos_numerico:
            atributos.append(keys[num_atb-1])

        if len(atributos) + len(lista_ids_label) == len(keys):
            todos = True  # se forem usadas todas as colunas, será indicado
    else:
        todos = True
        for atributo in keys:
            if atributo not in lista_ids_label:
                atributos.append(atributo) # atributos usados serão todos menos os que compõem o id

    if lista_restricoes_numerico != ["none"]:
        lista_restricoes = []
        for num_rest in lista_restricoes_numerico:
            lista_restricoes.append(keys[num_rest-1])
    else:
        lista_restricoes = lista_restricoes_numerico

    # == Prints para mostrar parâmetros selecionados ==

    print("Atributos usados:")
    print(f"Lista de atributos: {atributos}")
    print(f"Lista de atributos para ids: {lista_ids_label}")
    print(f"Lista de atributos restritivos: {lista_restricoes}")
    print(f"Lista de medidas de centralidade: {lista_medidas}")
    output_m = "True" if lista_medidas != ["none"] else "False"
    print(f"Tomar medidas: {output_m}")
    print(f"Limiar: {limiar}")
    print(f"Usar lógica or: {usar_or}")
    print(f"Arquivo: {nome_arquivo_csv}")
    print(f"Gerar imagem do grafo: {nao_gerar_imagem_grafo}")
    print(f"Usar grafo puro: {usar_grafo_puro}")
    print(f"Mostrar apenas giant component: {giant_component}")
    print(f"Não normalizar entrada: {use_raw_data}")
    print(f"Mostra vértices com degree a partir de: {min_degree}")
    print(f"Step a partir do qual os vértices irão compor o grafo: {min_step}")
    print("")

    # == Cria ids ==

    print("Gerando lista de ids...")
    lista_ids = cria_lista_ids(lista_dict, lista_ids_label) # monta lista de identificadores dos vértices do grafo
    if not ids_validos(lista_ids):  # se os ids gerados não forem únicos
        print("Erro: ids gerados não são únicos, usar outros atributos")
        sys.exit("Saindo do programa")
    else:
        print("Ids válidos")

    # == Monta lista de arestas ==

    print("Gerando arestas...")
    arestas, pesos_arestas = monta_arestas(atributos, lista_dict, lista_restricoes, usar_or, limiar, precisao)

    # == Cria grafo e o processa ==

    print("Atribuindo valores ao grafo...")
    g_raw = ig.Graph()
    n_vertices = (len(lista_dict))
    g_raw.add_vertices(n_vertices)
    g_raw.vs["label"] = lista_ids  # label do grafo é a lista de ids
    g_raw.add_edges(arestas)  # grafo recebe as arestas
    g_raw.es["peso"] = pesos_arestas  # arestas recebem seus pesos

    for key in keys:
        g_raw.vs[key] = [veiculo[key] for veiculo in lista_dict] # grafo recebe os atributos dos dicionários

    # pega o nome do atributo referente ao step no arquivo de entrada
    nome_step = list(filter(lambda x: x == "Step" or x == "step", keys))[0]

    g = g_raw.copy()  # copia o grafo original
    to_delete_vertices = []
    # verifica se o usuário escolheu remover os vértices cujo grau é zero
    if not usar_grafo_puro:
        if min_step > 0:  # se for considerado um step mínimo, remover vértices abaixo desse step
            for v in g.vs:
                if v[nome_step] < min_step:
                    to_delete_vertices.append(v) # seleciona ids com step abaixo do mínimo
            g.delete_vertices(to_delete_vertices) # remove vértices com step abaixo do mínimo

        to_delete_vertices = [] # remove vértices cujo grau é zero
        for v in g.vs:
            if v.degree() == 0:
                to_delete_vertices.append(v) # seleciona ids com degree zero
        g.delete_vertices(to_delete_vertices) # remove todos os ids que não formam nenhuma aresta (cujo grau é zero)
    else:
        if min_step > 0: # se o usuário escolheu não remover os vértices de grau zero, verificar se escolheu um step mínimo
            for v in g.vs:
                if v[nome_step] < min_step:
                    to_delete_vertices.append(v)
            g.delete_vertices(to_delete_vertices)

    print("Pronto")  # finalizou a atribuição

    print("\nInformações do grafo gerado:") # mostra informações do grafo, como número de vértices e quantidade de arestas
    print(g.degree_distribution())
    print(g.summary())

    # == Trata custo computacional ==

    medidas_custosas = ["betweenness"] # lista de medidas que são custosas e não desejáveis de ser tomadas se o grafo for muito grande
    possui_medida_custosa = determina_possui_medida_custosa(lista_medidas, medidas_custosas)
    nova_lista_medidas = lista_medidas.copy() # usada para, se for escolhido, filtrar as medidas que são custosas
    custo = 0  # define a intensidade do custo computacional: 0 para baixo, 1 para médio e 2 para alto
    opcao_grafo_grande = 0

    if g.ecount() <= arestas_para_custoso:
        custo = 0
    else:
        custo = 1

    # se o usuário optou por gerar uma imagem do grafo ou realizar alguma medida
    if not nao_gerar_imagem_grafo or lista_medidas != ["none"]:
        if custo == 1:
            if possui_medida_custosa:
                print(f"O grafo possui mais que {arestas_para_custoso} arestas. O custo computacional para gerar uma imagem do grafo e tomar medidas de centralidade custosas será alto. Você deseja tomá-las e gerar uma imagem do grafo?")
                print("1 - Tomar medidas custosas e gerar imagem do grafo")
                print("2 - Apenas tomar medidas custosas")
                print("3 - Apenas gerar imagem do grafo")
                print("4 - Não gerar imagem e não tomar medidas custosas")

                # recebe o input do usuário, verificando a consistência da entrada
                while opcao_grafo_grande != 1 and opcao_grafo_grande != 2 and opcao_grafo_grande != 3 and opcao_grafo_grande != 4:
                    opcao_grafo_grande = int(input("Digite sua opção: "))
                    if opcao_grafo_grande != 1 and opcao_grafo_grande != 2 and opcao_grafo_grande != 3 and opcao_grafo_grande != 4:
                        print("Opção inválida, digite novamente.")
            else:
                print(f"O grafo possui mais que {arestas_para_custoso} arestas. O custo computacional para gerar uma imagem do grafo será alto. Você deseja gerar a imagem do grafo?")
                print("1 - Gerar imagem do grafo")
                print("2 - Não gerar imagem do grafo")

                # recebe o input do usuário, verificando a consistência da entrada
                while opcao_grafo_grande != 1 and opcao_grafo_grande != 2:
                    opcao_grafo_grande = int(input("Digite sua opção: "))
                    if opcao_grafo_grande != 1 and opcao_grafo_grande != 2:
                        print("Opção inválida, digite novamente.")

            # se for escolhido para não tomar medidas custosas
            if opcao_grafo_grande == 3 or opcao_grafo_grande == 4:
                nova_lista_medidas = []  # esvazia lista de medidas

                for medida in lista_medidas:
                    if medida not in medidas_custosas:
                        nova_lista_medidas.append(medida) # filtra medidas custosas da lista de medidas

        # salva informações que irão compor os nomes dos arquivos de saída
        nome_dados = monta_nome(g, limiar, lista_atributos_numerico)

    # == Gera imagem do grafo ==

    if not nao_gerar_imagem_grafo:
        # se foi selecionado para fazer a imagem do grafo, ou se não for custoso
        if opcao_grafo_grande == 1 or opcao_grafo_grande == 3 or custo == 0:
            print("\nPlotando grafo...")
            if g.vcount() != 0:
                create_directory("results")
                create_directory("results/graphs")
                nome_imagem_grafo = f"img_{nome_dados}.pdf"
                
                if giant_component:  # se foi escolhido para apenas mostrar o giant component do grafo
                    g_plot = g.components().giant().copy()
                else:  # caso contrario, mostrar todo o grafo
                    g_plot = g.copy()

                if min_degree < 0:
                    print("Degree mínimo é negativo e será desconsiderado")
                elif min_degree != 0:
                    while possui_grau_minimo(g_plot, min_degree):
                        to_delete_vertices = []
                        for v in g_plot.vs:
                            if v.degree() < min_degree: # apenas deixar vértices cujo grau é igual ou maior que o passado em mdeg
                                to_delete_vertices.append(v)
                        g_plot.delete_vertices(to_delete_vertices)

                if g_plot.vcount() != 0: # se o grafo não estiver vazio, plotar
                    visual_style = determine_visual_style(g_plot)
                    ig.plot(g_plot, target=f"results/graphs/{nome_imagem_grafo}", **visual_style)
                    print(f"Imagem {nome_imagem_grafo} gerada")
                else:
                    print("Nenhuma imagem será gerada, pois o grafo está vazio")
            else:
                print("Nenhuma imagem será gerada, pois o grafo está vazio")
        else:
            print("O grafo não será plotado.")

    # == Toma medidas de caracterização ==

    if nova_lista_medidas != ["none"]:
        print("Gerando tabela...")

        if len(nova_lista_medidas) != 0:
            if g.vcount() != 0:
                nome_tabela = f"table_{nome_dados}.pdf"
                nome_tabela_freq = f"freq_table_{nome_dados}.pdf"
                # tabela com as medidas de caracterização selecionadas é gerada
                monta_tabela(dados=calcula_medidas(g, nova_lista_medidas, g.vs["label"]), nome=nome_tabela, tipo="centrality")
                print(f"Tabela {nome_tabela} gerada")
                # tabela de frequências é gerada
                monta_tabela(dados=calculate_frequency_keys(g, "Link"), nome=nome_tabela_freq, tipo="frequency")
                print(f"Tabela {nome_tabela_freq} gerada")
            else:
                print("Nenhuma tabela será gerada, pois o grafo está vazio")
        else:
            print("Lista de medidas está vazia.")

    dict_vizinhos = cria_dicionario_vizinhos_links(g, keys, intervalo=250, max_step=calcula_max_step(lista_dict, keys))

    return dict_vizinhos