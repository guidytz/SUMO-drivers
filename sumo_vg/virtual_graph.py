# functions to create virtual graph

import datetime as dt
import itertools
import math
import sys
from collections import Counter
from csv import DictReader
from decimal import Decimal, getcontext
from pathlib import Path
import igraph as ig
import matplotlib.pyplot as plt
from pandas import DataFrame


# == Funções para as diferentes partes do programa ==


# - Importação de dados -


def eh_numero(num_str: str) -> bool:
    '''
    Le csv inteiro e o guarda na memória, após, salva cada linha da leitura em uma lista
    verifica se string é um númerico ou não 
    '''
    try:
        float(num_str)
    except ValueError:
        return False
    return True


def tem_atributo_vazio(linha: dict, keys: list) -> bool:
    '''
    Recebe um dicionário e as keys deste dicionário, verificando se existe nele alguma entrada vazia
    '''
    for key in keys:
        if linha[key] == "":
            return True
    return False


def has_zero_occupancy(row: dict) -> bool:
    '''
    Checks if row has occupancy of zero
    '''
    if float(row["Occupancy"]) == 0:
        return True
    else:
        return False


def is_border_link(link: str) -> bool:
    '''
    Checks if link is border link (only makes sense in grid network)
    '''
    if "top" in link or "bottom" in link or "right" in link or "left" in link:
        return True
    else:
        return False


def should_include_line(line: dict[str, list], keys: list, link_as_vertex: bool) -> bool:
    '''
    Checks if line should be considered or discarded. This line will become a vertex of the virtual graph.
    Doesn't include line containing empty attribute, zero occupancy or border links (if graph will be composed of links).
    '''
    if (link_as_vertex):
        if is_border_link(line["Link"]):
            return False
    if tem_atributo_vazio(line, keys) or has_zero_occupancy(line):
        return False
    return True

def should_normalize(key_value: list) -> bool:
    '''
    Determines if an imported column of the input csv should be normalized or not. 
    '''
    if eh_numero(str(key_value)):
        return True

    return False


def importa_csv(caminho_csv: str, first_id: int) -> list:  # recebe o caminho e o nome do arquivo a ser lido
    '''
    Lê os dados do csv e retorna cada linha como um dicionário cujas keys são o header do csv e uma lista com as keys do dicionário
    '''
    with open(caminho_csv) as arquivo:
        # lê cada atributo como string, que será convertido posteriormente
        leitura = DictReader(arquivo)
        keys = list(leitura.fieldnames or [])  # guardas as keys do dicionário

        attribute_name = keys[first_id-1]
        link_as_vertex = True if attribute_name == "Link" else False

        linhas = []
        id = 0  # identificador usado para criar arestas
        num_linha = 2
        for linha in leitura:
            if should_include_line(linha, keys, link_as_vertex):
                linhas.append(linha)  # grava cada linha (um dicionário próprio) em uma lista de linhas
                linha["id"] = id
                id += 1
            num_linha += 1

        # converte atributos do dicionário para seus tipos respectivos (inicialmente são strings)
        for linha in linhas:
            for key in keys:
                if eh_numero(linha[key]):
                    linha[key] = float(linha[key])  # converte os atributos numéricos para float

    # retorna lista contendo dicionários e outra lista com as keys desses dicionários (são todas as mesmas)
    return linhas, keys


def normaliza_lista(lista_atributos: list) -> list: 
    '''
    Normaliza os valores de uma lista (com a fórmula n = (x-min)/(max-min)) e retorna uma lista com valores normalizados
    '''
    minimo = min(lista_atributos)
    maximo = max(lista_atributos)

    lista_atributos_norm = []

    if (maximo-minimo) == 0:
        exit("Erro na normalização: divisão por zero")

    for atributo in lista_atributos:
        lista_atributos_norm.append((atributo-minimo)/(maximo-minimo))

    return lista_atributos_norm


def normaliza_lista_dict(lista_dicionarios: list, keys: list, lista_ids_label: list) -> list:
    '''
    Normaliza os valores de uma lista de dicionários
    '''
    lista_dicionarios_norm = lista_dicionarios
    keys_norm = []
    for key in keys:
        if key not in lista_ids_label and should_normalize(lista_dicionarios[0][key]):  # só os atributos que são númericos e não compõem o label serão normalizados
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


def cria_lista_ids(lista_dicionarios: list, atributos_usados: list) -> list:
    '''
    Recebe a lista contendo os vértices do grafo e os nomes dos atributos que irão compor o id, retornando a lista com os atributos concatenados
    '''
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


def ids_validos(lista_ids: list) -> bool:
    '''
    Recebe a lista contendo os ids dos nodos e retorna True se todos forem diferentes ou False caso contrário
    '''
    pares_ids = itertools.combinations(lista_ids, 2)  # cria pares de ids

    for par in pares_ids:
        if par[0] == par[1]:  # se qualquer par for igual, retorna False
            return False

    return True  # caso contrário, retorna True


def cria_string_com_atributos(lista_atributos: list) -> str:
    '''
    Recebe uma lista de atributos e monta uma string com estes atributos separados por "-"
    '''
    string_atributos = ""
    for i in range(len(lista_atributos)):
        if i == len(lista_atributos)-1:
            string_atributos += str(lista_atributos[i])
        else:
            string_atributos += f"{str(lista_atributos[i])}-"
    return string_atributos


def gets_name_file(directory_file: str) -> str:
    '''
    Input: directory and name of file
    Output: str containing the name of the file
    '''
    path = Path(directory_file)
    if path.suffix == ".csv":
        return path.stem
    else:
        return directory_file


def monta_nome(limiar: float, lista_atributos: list, directory_file: str) -> str:
    '''
    Recebe os parâmetros do usuário e gera o nome do arquivo que contém alguns dados do grafo
    '''
    tempo = dt.datetime.now()
    hora_atual = tempo.strftime('%H%M%S')
    atributos = cria_string_com_atributos(lista_atributos)
    limiar_str = str(limiar)
    # remove o ponto do limiar para nao causar problemas com o nome e a extensão do arquivo
    limiar_str_processada = limiar_str.replace('.', "'")
    directory_file_stem = gets_name_file(directory_file)
    nome_final = f"{hora_atual}_atb{atributos}_l{limiar_str_processada}_{directory_file_stem}"

    return nome_final


def converte_intervalo(intervalo: str) -> list:
    '''
    Converte string representando intervalo numérico na forma "início-fim" em uma lista contendo os números naquele intervalo
    '''
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


def processa_int_ou_intervalo(entrada: list | str) -> list:
    '''
    Determina se a entrada é um intervalo numérico na forma "início-fim" ou uma lista de inteiros, retornando a lista de inteiros que corresponde ao intervalo ou a própria lista de inteiros
    '''
    lista_intervalo = []

    for v in entrada:
        if "-" in v:
            for num in converte_intervalo(v):
                lista_intervalo.append(num)
        else:
            lista_intervalo.append(int(v))  # transforma os números em inteiros

    return lista_intervalo


# - Criação de arestas -


def verifica_aresta(lista_resultado: list, usar_or: bool) -> bool:
    '''
    Dependendo da lógica escolhida, verifica se deve ser criada uma aresta entre um par de nodos
    '''
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


def dentro_restricao(v1: dict, v2: dict, lista_restricoes: list | None) -> bool:
    '''
    Dados dois vértices e uma lista de atributos usados como restrição, a função retorna -1 se a lista de restrições contiver "None",
    False se os vértices possuírem os mesmos valores para os mesmos atributos restritivos ou True se possuírem todos os valores diferentes
    para os mesmos atributos restritivos
    '''
    if lista_restricoes is not None:
        for restricao in lista_restricoes:
            if v1[restricao] == v2[restricao]:
                return False
    return True


def dentro_limiar(v1: dict, v2: dict, atributo: str, limiar: float, precisao: float) -> bool:
    '''
    Verifica se um atributo entre dois dicionários está dentro do limiar ou não
    recebe dois dicionários, um atributo, um limiar e a precisão da diferença entre atributos
    '''
    getcontext().prec = precisao
    return abs(Decimal(v1[atributo]) - Decimal(v2[atributo])) <= limiar


def monta_arestas(atributos: list, lista_dicionarios: list, lista_restricoes: list, usar_or: bool, limiar: float, precisao: float):
    '''
    Monta uma lista de arestas a partir de uma lista de atributos, uma de dicionários, uma de restrições, uma lógica para montar arestas e um limiar
    '''
    arestas = []
    pesos_arestas = []
    # para cada par de dicionários da lista
    for v1, v2 in itertools.combinations(lista_dicionarios, 2):
        # indica se para cada atributo, deve haver uma aresta (1) ou não (0)
        lista_resultados = []
        if dentro_restricao(v1, v2, lista_restricoes):
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

    # retorna lista com arestas e lista com pesos das arestas
    return arestas, pesos_arestas


# - Toma medidas sobre o grafo -


def determina_possui_medida_custosa(lista_medidas: list | None, medidas_custosas: list) -> bool:
    '''
    Determina se lista de medidas possui alguma medida considerada custosa
    '''
    if lista_medidas is not None:
        for medida_custosa in medidas_custosas:
            if medida_custosa in lista_medidas:
                return True
    return False


def calcula_medidas(grafo: ig.Graph, lista_medidas: list, lista_ids: list) -> dict:
    '''
    Calcula medidas de centralidade do grafo, retornando um dicionário com as medidas
    '''
    dict_valores_medidas: dict[str, list] = {}
    for medida in lista_medidas:
        lista_medida_calculada = getattr(grafo, medida)()

        lista_ids_num_ordenada = []
        if medida == lista_medidas[0]:
            lista_medida = lista_medida_calculada.copy()  # usada para ordenar tabela
            lista_medida.sort(reverse=True)

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


def calculate_frequency_keys(graph: ig.Graph, attribute: str) -> dict:
    '''
    Input: graph representing network composed of nodes that are dictionaries
    Output: dictionary with the frequency that the specified key appears in the graph
    '''
    list_keys = []
    for v in graph.vs:
        list_keys.append(v[attribute])

    dict_freq_keys = dict(Counter(key for key in list_keys))

    # creates sorted dictionary by value in key in decreasing order
    sorted_dict_freq_keys = {item[0]: item[1] for item in sorted(dict_freq_keys.items(), key=lambda x: (-x[1], x[0]))}

    processed_dict = dict()
    processed_dict[attribute] = sorted_dict_freq_keys.keys()

    frequency_list = []
    for item in list(sorted_dict_freq_keys.items()):
        frequency_list.append(item[1])

    processed_dict["frequency"] = frequency_list

    return processed_dict


def monta_tabela(dados: list, nome: str, tipo: str) -> None:
    '''
    Recebe os dados em uma lista, o nome que o arquivo de saída terá e se o dados se referem 
    às medidas de centralidade ("centrality") ou frequência ("frequency")
    '''
    local_path = Path(f"results/tables/{tipo}")
    local_path.mkdir(exist_ok=True, parents=True)
    local_nome = Path(nome)

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.axis("tight")

    colunas = dados
    df = DataFrame(colunas, columns=colunas)

    ax.table(cellText=df.values, colLabels=list(df.columns), cellLoc="center", loc="upper center")

    plt.savefig(f"{str(local_path/local_nome)}", format="pdf", bbox_inches="tight")

    print(f"Table '{local_nome}' generated at {local_path}")


# - Visual do grafo -


def calcula_bbox(n_arestas: int) -> tuple:
    '''
    Calcula o tamanho da imagem do grafo
    '''
    bbox_limit = 3000

    # criada com raiz quadrada, chegando perto dos pontos
    bbox = math.sqrt(n_arestas)*150

    if bbox > bbox_limit:  # define limite para bbox
        bbox = bbox_limit

    return (bbox, bbox)


def determina_cor_vertice(grau: float, media: float) -> str:
    '''
    Determina cor de um vértice, quão maior o degree, mais quente a cor
    '''
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


def lista_cores(g: ig.Graph) -> list:
    '''
    Cria lista de cores dos vértices do grafo
    '''
    lista_graus = g.degree()

    if len(lista_graus) == 0:
        media_graus = 0
    else:
        media_graus = sum(lista_graus) / len(lista_graus)

    lista_cores = []
    for grau in lista_graus:
        lista_cores.append(determina_cor_vertice(grau, media_graus))

    return lista_cores


def determine_visual_style(g: ig.Graph) -> dict:
    '''
    Determina características visuais do grafo
    '''
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
    visual_style["vertex_label_size"] = [
        max(10, 4*math.sqrt(d) if 4*math.sqrt(d) < font_lower_bound else font_lower_bound) for d in g.degree()]
    visual_style["layout"] = g.layout("auto")

    return visual_style


def possui_grau_minimo(grafo: ig.Graph, d_minimo: float) -> bool:
    '''
    Determina se o grafo possui algum vértice com grau mínimo passado
    '''
    for v in grafo.vs:
        if v.degree() < d_minimo:
            return True
    return False


def calcula_max_step(lista_dict: list, keys: list) -> int:
    '''
    Calcula step máximo 
    '''
    nome_step = list(filter(lambda x: x == "Step" or x == "step", keys))[0]

    lista_step = []
    for v in lista_dict:
        lista_step.append(v[nome_step])

    return int(max(lista_step))


def verifica_se_esta_no_intervalo(limite_inferior: int, limite_superior: int, valor: float, ultimo_intervalo: bool) -> bool:
    '''
    Recebe número e verifica se este está dentro de um intervalo [x, y) (ou, se for o último intervalo, [w, z])
    '''
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


def retorna_vizinhos(list_attribute_vertices: list) -> list:
    '''
    Recebe uma lista de vértices com mesmo atributo especificado e retorna uma lista contendo todos os vizinhos desse atributo
    '''
    list_attribute_neighbors = []
    for attribute_v in list_attribute_vertices:
        list_vertex_neighbors = attribute_v.neighbors()
        for neighbor_v in list_vertex_neighbors:
            if neighbor_v not in list_attribute_neighbors:
                list_attribute_neighbors.append(neighbor_v)

    return list_attribute_neighbors


def retorna_vizinhos_no_intervalo(list_neighbors: list, intervalo: tuple, ultimo_intervalo: bool, nome_step: str, attribute_name: str) -> list:
    '''
    Recebe uma lista de vizinhos do atributo especificado e filtra os vértices que estão no intervalo passado
    '''
    limite_inferior = intervalo[0]
    limite_superior = intervalo[1]

    # para cada vertice, verificar se esta no intervalo e adicionar à lista
    lista_vizinhos_no_intervalo = []
    for vizinho in list_neighbors:
        if verifica_se_esta_no_intervalo(limite_inferior, limite_superior, vizinho[nome_step], ultimo_intervalo):
            if vizinho[attribute_name] not in lista_vizinhos_no_intervalo:
                lista_vizinhos_no_intervalo.append(vizinho[attribute_name])

    return lista_vizinhos_no_intervalo


def cria_dicionario_vizinhos(grafo: ig.Graph, keys: list, attribute_name: str , intervalo: int, max_step: int) -> dict:
    '''
    Cria um dicionário contendo todos os vizinhos em determinado intervalo de tempo do grafo agrupados por um atributo (ex. link ou junction).
    '''
    dict_vizinhos = dict()
    nome_step = list(filter(lambda x: x == "Step" or x == "step", keys))[0]

    list_attributes = []
    for v in grafo.vs:
        list_attributes.append(v[attribute_name])

    unique_attributes_list = []
    for attribute in list_attributes:
        if attribute not in unique_attributes_list:
            unique_attributes_list.append(attribute)

    # constroi lista de intervalos (compostos por uma tupla) utilizando o último step e o tamanho do intervalo selecionado: ["[0-1)", "[1-2) ... [n-1, n]"]
    lista_intervalos = []
    # obs: intervalo é fechado à esquerda e aberto à direita, mas o último intervalo é fechado nos dois lados
    nome_intervalo = (0, 0)
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
    
    for attribute in unique_attributes_list:
        dict_vizinhos[attribute] = dict()

        # lista de vértices do grafo com atributo especificado
        list_attribute_vertices = list(filter(lambda v: v[attribute_name] == attribute, grafo.vs))

        for i in range(num_intervalos):
            # se for o último intervalo, ele será fechado dos dois lados
            ultimo_intervalo = True if i == num_intervalos-1 else False

            lower_interval = lista_intervalos[i][0]
            upper_interval = lista_intervalos[i][1]

            dict_vizinhos[attribute][lista_intervalos[i]] = [] # empties list of neighbors for that interval
            v_neighbors_at_interval = []
            for v_attribute in list_attribute_vertices:
                if verifica_se_esta_no_intervalo(lower_interval, upper_interval, v_attribute[nome_step], ultimo_intervalo):
                    v_neighbors_at_interval = v_attribute.neighbors() # gets neighbors at interval
                for v_neighbor in v_neighbors_at_interval:
                    if v_neighbor[attribute_name] not in dict_vizinhos[attribute][lista_intervalos[i]]:
                        dict_vizinhos[attribute][lista_intervalos[i]].append(v_neighbor[attribute_name]) # filters and appends list

    return dict_vizinhos


def generate_graph_neighbors_dict(nome_arquivo_csv: str, lista_atributos_numerico: list, lista_ids_label_numerico: list, lista_restricoes_numerico: list | None,
                                   limiar: float, usar_or: bool, lista_medidas: list | None, nao_gerar_imagem_grafo: bool, usar_grafo_puro: bool, giant_component: bool,
                                   not_normalize: bool, min_degree: int, min_step: int, arestas_para_custoso: int, precisao: int, intervalo_vizinhos: int,
                                   network_name: str) -> dict:
    '''
    Main script to generate the virtual graph, its image, take centrality measurements of it and generate 
    finally the virtual graph neighbors dictionary
    '''

    # == Processa listas numéricas ==

    if lista_atributos_numerico != ["ALL"]:
        lista_atributos_numerico = processa_int_ou_intervalo(lista_atributos_numerico)

    lista_ids_label_numerico = processa_int_ou_intervalo(lista_ids_label_numerico)

    if lista_restricoes_numerico is not None:
        lista_restricoes_numerico = processa_int_ou_intervalo(lista_restricoes_numerico)

    # == Lê csv, traduz entrada numérica dos ids para atributos e normaliza dados, se foi pedido ==

    print("Reading file...")
    lista_dict, keys = importa_csv(nome_arquivo_csv, lista_ids_label_numerico[0])

    lista_ids_label = []  # usada como label do grafo, indica também atributos que não serão normalizados
    for num_id in lista_ids_label_numerico:  # traduz os número passados como argumento correspondente às colunas
        lista_ids_label.append(keys[num_id-1])  # numeração das colunas começa em 1, por isso -1

    if not not_normalize:
        lista_dict, keys = normaliza_lista_dict(lista_dict, keys, lista_ids_label)
    print("File read.")

    # Traduz entrada numérica dos outros parâmetros

    print("Translating atributes...")

    atributos = []
    if lista_atributos_numerico != ["ALL"]:
        for num_atb in lista_atributos_numerico:
            atributos.append(keys[num_atb-1])
    else:
        for atributo in keys:
            if atributo not in lista_ids_label:
                atributos.append(atributo)  # atributos usados serão todos menos os que compõem o id

    if lista_restricoes_numerico is not None:
        lista_restricoes = []
        for num_rest in lista_restricoes_numerico:
            lista_restricoes.append(keys[num_rest-1])
    else:
        lista_restricoes = lista_restricoes_numerico

    vertex_attribute_name = keys[lista_ids_label_numerico[0]-1]

    # == Prints para mostrar parâmetros selecionados ==

    print(f"Attributes: {atributos}")
    print(f"Labels: {lista_ids_label}")
    print(f"Restrictions: {lista_restricoes}")
    print(f"Centrality measures: {lista_medidas}")
    output_m = "True" if lista_medidas is not None else "False"
    print(f"Take centrality measures: {output_m}")
    print(f"Limiar: {limiar}")
    print(f"Use or logic: {usar_or}")
    print(f"File: {nome_arquivo_csv}")
    print(f"No virtual graph image: {nao_gerar_imagem_grafo}")
    print(f"Use pure virtual graph: {usar_grafo_puro}")
    print(f"Only plot giant component: {giant_component}")
    print(f"Don't normalize input: {not_normalize}")
    print(f"Plots vertices with a degree bigger or equal to: {min_degree}")
    print(f"Plots vertices with a step bigger or equal to: {min_step}")
    print(f"Amplitude of timestep of virtual graph neighbors dictionary: {intervalo_vizinhos} steps")
    print(f"Virtual graph's vertices: {vertex_attribute_name}")

    # == Cria ids ==

    print("Generating labels...")
    lista_ids = cria_lista_ids(lista_dict, lista_ids_label)  # monta lista de identificadores dos vértices do grafo
    if not ids_validos(lista_ids):  # se os ids gerados não forem únicos
        print("Error! Labels created aren't unique. Use other atributes")
        sys.exit("Exiting program")
    else:
        print("Labels are valid")

    # == Monta lista de arestas ==

    print("Generating edges...")
    arestas, pesos_arestas = monta_arestas(atributos, lista_dict, lista_restricoes, usar_or, limiar, precisao)

    # == Cria grafo e o processa ==

    print("Atributing values to the virtual graph...")
    g_raw = ig.Graph()
    n_vertices = (len(lista_dict))
    g_raw.add_vertices(n_vertices)
    g_raw.vs["label"] = lista_ids  # label do grafo é a lista de ids
    g_raw.add_edges(arestas)  # grafo recebe as arestas
    g_raw.es["peso"] = pesos_arestas  # arestas recebem seus pesos

    for key in keys:
        g_raw.vs[key] = [veiculo[key] for veiculo in lista_dict]  # grafo recebe os atributos dos dicionários

    # pega o nome do atributo referente ao step no arquivo de entrada
    nome_step = list(filter(lambda x: x == "Step" or x == "step", keys))[0]

    g = g_raw.copy()  # copia o grafo original
    to_delete_vertices = []
    # verifica se o usuário escolheu remover os vértices cujo grau é zero
    if not usar_grafo_puro:
        if min_step > 0:  # se for considerado um step mínimo, remover vértices abaixo desse step
            for v in g.vs:
                if v[nome_step] < min_step:
                    to_delete_vertices.append(v)  # seleciona ids com step abaixo do mínimo
            g.delete_vertices(to_delete_vertices)  # remove vértices com step abaixo do mínimo

        to_delete_vertices = []  # remove vértices cujo grau é zero
        for v in g.vs:
            if v.degree() == 0:
                to_delete_vertices.append(v)  # seleciona ids com degree zero
        g.delete_vertices(to_delete_vertices)  # remove todos os ids que não formam nenhuma aresta (cujo grau é zero)
    else:
        if min_step > 0:  # se o usuário escolheu não remover os vértices de grau zero, verificar se escolheu um step mínimo
            for v in g.vs:
                if v[nome_step] < min_step:
                    to_delete_vertices.append(v)
            g.delete_vertices(to_delete_vertices)

    print("Done")  # finalizou a atribuição

    # mostra informações do grafo, como número de vértices e quantidade de arestas
    print("Information about the virtual graph:")
    print(g.degree_distribution())
    print(g.summary())

    # == Trata custo computacional ==

    # lista de medidas que são custosas e não desejáveis de ser tomadas se o grafo for muito grande
    medidas_custosas = ["betweenness"]
    possui_medida_custosa = determina_possui_medida_custosa(lista_medidas, medidas_custosas)
    nova_lista_medidas = lista_medidas.copy() if lista_medidas is not None else [] # usada para, se for escolhido, filtrar as medidas que são custosas
    custo = 0  # define a intensidade do custo computacional: 0 para baixo, 1 para médio e 2 para alto
    opcao_grafo_grande = 0
    nome_dados = ""

    if g.ecount() <= arestas_para_custoso:
        custo = 0
    else:
        custo = 1

    # se o usuário optou por gerar uma imagem do grafo ou realizar alguma medida
    if not nao_gerar_imagem_grafo or lista_medidas is not None:
        if custo == 1:
            if possui_medida_custosa:
                print(
                    f"The graph has more than {arestas_para_custoso} edges. Do you really wish to generate an image of this graph and take costly centrality measures?")
                print("1 - Take measures and generate image")
                print("2 - Only take measures")
                print("3 - Only genrate image")
                print("4 - Don't generate image neither take measures")

                # recebe o input do usuário, verificando a consistência da entrada
                while opcao_grafo_grande != 1 and opcao_grafo_grande != 2 and opcao_grafo_grande != 3 and opcao_grafo_grande != 4:
                    opcao_grafo_grande = int(input("Type your option: "))
                    if opcao_grafo_grande != 1 and opcao_grafo_grande != 2 and opcao_grafo_grande != 3 and opcao_grafo_grande != 4:
                        print("Invalid option, type again.")
            else:
                print(
                    f"The graph has more than {arestas_para_custoso} edges. Do you really wish to generate an image of this graph?")
                print("1 - Yes")
                print("2 - No")

                # recebe o input do usuário, verificando a consistência da entrada
                while opcao_grafo_grande != 1 and opcao_grafo_grande != 2:
                    opcao_grafo_grande = int(input("Type your option: "))
                    if opcao_grafo_grande != 1 and opcao_grafo_grande != 2:
                        print("Invalid option, type again.")

            # se for escolhido para não tomar medidas custosas
            if opcao_grafo_grande == 3 or opcao_grafo_grande == 4:
                nova_lista_medidas = []  # esvazia lista de medidas

                for medida in lista_medidas:
                    if medida not in medidas_custosas:
                        nova_lista_medidas.append(medida)  # filtra medidas custosas da lista de medidas

        # salva informações que irão compor os nomes dos arquivos de saída
        nome_dados = monta_nome(limiar, lista_atributos_numerico, network_name)

    # == Gera imagem do grafo ==

    if not nao_gerar_imagem_grafo:
        # se foi selecionado para fazer a imagem do grafo, ou se não for custoso
        if opcao_grafo_grande == 1 or opcao_grafo_grande == 3 or custo == 0:
            print("Ploting virtual graph...")
            if g.vcount() != 0:
                vg_path = Path("results/graphs")
                vg_path.mkdir(exist_ok=True, parents=True)
                vg_name = Path(f"img_{nome_dados}.pdf")

                if giant_component:  # se foi escolhido para apenas mostrar o giant component do grafo
                    g_plot = g.components().giant().copy()
                else:  # caso contrario, mostrar todo o grafo
                    g_plot = g.copy()

                if min_degree < 0:
                    print("Minimum degree is negative and will not be considered")
                elif min_degree != 0:
                    while possui_grau_minimo(g_plot, min_degree):
                        to_delete_vertices = []
                        for v in g_plot.vs:
                            if v.degree() < min_degree:  # apenas deixar vértices cujo grau é igual ou maior que o passado em mdeg
                                to_delete_vertices.append(v)
                        g_plot.delete_vertices(to_delete_vertices)

                if g_plot.vcount() != 0:  # se o grafo não estiver vazio, plotar
                    visual_style = determine_visual_style(g_plot)
                    ig.plot(g_plot, target=str(vg_path/vg_name), **visual_style)
                    print(f"Image '{str(vg_name)}' generated at {vg_path}")
                else:
                    print("Empty virtual graph, no image generated")
            else:
                print("Empty virtual graph, no image generated")
        else:
            print("The virtual graph will not be ploted")

    # == Toma medidas de caracterização ==

    if len(nova_lista_medidas) > 0:
        print("Generating table...")

        if len(nova_lista_medidas) != 0:
            if g.vcount() != 0:
                nome_tabela = f"table_{nome_dados}.pdf"
                nome_tabela_freq = f"freq_table_{nome_dados}.pdf"
                # tabela com as medidas de caracterização selecionadas é gerada
                monta_tabela(dados=calcula_medidas(g, nova_lista_medidas,
                             g.vs["label"]), nome=nome_tabela, tipo="centrality")
                # tabela de frequências é gerada
                monta_tabela(dados=calculate_frequency_keys(g, attribute=vertex_attribute_name), nome=nome_tabela_freq, tipo="frequency")
            else:
                print("Empty graph, no table generated")
        else:
            print("Centrality measurements list is empty")

    dict_vizinhos = cria_dicionario_vizinhos(
        g, keys, vertex_attribute_name, intervalo_vizinhos, max_step=calcula_max_step(lista_dict, keys))

    print("")

    for link, intervals in dict_vizinhos.items():
        print(link)
        for interval, neighbors in intervals.items():
            if len(neighbors) > 0:
                print(f"{interval}: {neighbors}")

    return dict_vizinhos
