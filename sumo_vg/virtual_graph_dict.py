# standalone script to generate virtual graph dictionary file

import argparse as ap
import pickle
import sys
import time
from pathlib import Path
import igraph as ig
from sumo_vg.virtual_graph import *

# == Variáveis globais ==
arestas_para_custoso = 2000  # quantidade de arestas para que o grafo seja considerado custoso
precisao = 10  # precisao do resultado da subtração de atributos dos vértices para comparação com limiar

def main():
    t_inicio = time.time()  # inicia temporizador de saída

    # == Argparse e argumentos do usuário ==

    parser = ap.ArgumentParser()
    parser.add_argument("-f", "--vg_file",
                        help="Path and name to the file containing the data that is going to be used to create the virtual graph.")

    parser.add_argument("-atb", "--vg_attributes", default=["ALL"], nargs="+",
                        help="List of atributes used to create the virtual graph. Atribute is given by the number of the column of the input file. (default = ['ALL'])")

    parser.add_argument("-id", "--vg_label", nargs="+",
                        help="List of atributes that will compose the label of the virtual graph. Atribute is given by the number of the column of the input file.")

    parser.add_argument("-rst", "--vg_restrictions", default=["none"], nargs="+",
                        help="List of atributes that the nodes cannot share in order to create an edge in the virtual graph. Atribute is given by the number of the column of the input file. (default = ['none'])")
    
    parser.add_argument("-tsh", "--vg_threshold", type=float, default=0,
                        help="Threshold used to create an edge in the virtual graph. (default = 0)")
    
    parser.add_argument("-o", "--use_or_logic", action="store_true", default=False,
                        help="Use or logic instead of the and logic to create an edge between nodes given multiple atributes. (default = false)")
    
    parser.add_argument("-ms", "--centrality_measures",  default=["none"], nargs="+",
                        help="List of centrality measures to be taken of the virtual graph. (default = none)")
    
    parser.add_argument("-ni", "--no_image", action="store_true", default=False,
                        help=f"Determines if an image of the virtual graph will not be generated. (default = false)")
    
    parser.add_argument("-rgraph", "--raw_graph", action="store_true", default=False,
                        help="Determines if all nodes with degree zero will not be removed. (default = false)")
    
    parser.add_argument("-giant", "--giant_component", action="store_true", default=False,
                        help="Determines if only the giant component of the virtual graph will be present in the virtual graph image. (default = false)")
    
    parser.add_argument("-not-norm", "--vg_not_normalize", action="store_true", default=False,
                        help="Determines if the input data will not be normalized. (default = false)")
    
    parser.add_argument("-mdeg", "--min_degree", type=int, default=0,
                        help="Only vertices with a degree bigger or equal to this value will be ploted. (default = 0)")
    
    parser.add_argument("-mstep", "--vg_min_step", type=int, default=0,
                        help="Only vertices with a step bigger or equal to this value will be ploted. (default = 0)")
    
    parser.add_argument("-int", "--interval", type=int, default=250,
                        help="Amplitude of the timestep interval of the virtual graph neighbours dictionary. (default = 250)")

    args = parser.parse_args()

    nome_arquivo_csv = args.vg_file  # nome do arquivo csv a ser usado para gerar grafo
    lista_atributos_numerico = args.vg_attributes  # lista de atributos, passados como número da coluna
    lista_ids_label_numerico = args.vg_label  # lista de ids usados no label, passados como número da coluna
    lista_restricoes_numerico = args.vg_restrictions  # lista de restricoes para criar arestas, passadas como número da coluna
    limiar = args.vg_threshold  # limiar usado para criar arestas
    usar_or = args.use_or_logic  # lógica para criar arestas
    lista_medidas = args.centrality_measures  # lista de medidas que serão tomadas do grafo
    nao_gerar_imagem_grafo = args.no_image  # define se será gerada uma imagem do grafo ou não
    # define se será usado o grafo sem processamento (remover vértices de grau zero) ou não
    usar_grafo_puro = args.raw_graph
    giant_component = args.giant_component  # define se apenas o giant component será mostrado na imagem
    use_raw_data = args.vg_not_normalize  # define se os dados usados serão normalizados
    min_degree = args.min_degree  # apenas serão mostrados vértices com grau a partir do especificado
    min_step = args.vg_min_step  # apenas serão considerados vértices cujo step é maior ou igual a este valor
    interval_amplitude = args.interval  # amplitude of the virtual graph neighbours dictionary

    # == Verfica consistência de entrada ==

    if args.vg_label == None:
        print("Error! Labels parameter wasn't informed!")
        sys.exit("Exiting program")

    if args.vg_file == None:
        print("Error! Input file path and name wasn't informed")
        sys.exit("Exiting program")

    print("Parameters OK")  # se todos os parâmetros necessário foram informados

    # == Processa listas numéricas ==

    if lista_atributos_numerico != ["ALL"]:
        lista_atributos_numerico = processa_int_ou_intervalo(lista_atributos_numerico)

    lista_ids_label_numerico = processa_int_ou_intervalo(lista_ids_label_numerico)

    if lista_restricoes_numerico != ["none"]:
        lista_restricoes_numerico = processa_int_ou_intervalo(lista_restricoes_numerico)

    # == Lê csv, traduz entrada numérica dos ids para atributos e normaliza dados, se foi pedido ==

    print("Reading file...")
    lista_dict, keys = importa_csv(nome_arquivo_csv)

    lista_ids_label = []  # usada como label do grafo, indica também atributos que não serão normalizados
    for num_id in lista_ids_label_numerico:  # traduz os número passados como argumento correspondente às colunas
        lista_ids_label.append(keys[num_id-1])  # numeração das colunas começa em 1, por isso -1

    if not use_raw_data:
        lista_dict, keys = normaliza_lista_dict(lista_dict, keys, lista_ids_label)
    print("File read.")

    # Traduz entrada numérica dos outros parâmetros

    print("Translating atributes...")

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
                atributos.append(atributo)  # atributos usados serão todos menos os que compõem o id

    if lista_restricoes_numerico != ["none"]:
        lista_restricoes = []
        for num_rest in lista_restricoes_numerico:
            lista_restricoes.append(keys[num_rest-1])
    else:
        lista_restricoes = lista_restricoes_numerico

    # == Prints para mostrar parâmetros selecionados ==

    print("Atributes used:")
    print(f"Atributes: {atributos}")
    print(f"Labels: {lista_ids_label}")
    print(f"Restrictions: {lista_restricoes}")
    print(f"Centrality measures: {lista_medidas}")
    output_m = "True" if lista_medidas != ["none"] else "False"
    print(f"Take centrality measures: {output_m}")
    print(f"Limiar: {limiar}")
    print(f"Use or logic: {usar_or}")
    print(f"File: {nome_arquivo_csv}")
    print(f"No virtual graph image: {nao_gerar_imagem_grafo}")
    print(f"Use pure virtual graph: {usar_grafo_puro}")
    print(f"Only plot giant component: {giant_component}")
    print(f"Don't normalize input: {use_raw_data}")
    print(f"Plots vertices with a degree bigger or equal to: {min_degree}")
    print(f"Plots vertices with a step bigger or equal to: {min_step}")
    print(f"Amplitude of timestep of virtual graph neighbours dictionary: {interval_amplitude} steps")

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
    nova_lista_medidas = lista_medidas.copy()  # usada para, se for escolhido, filtrar as medidas que são custosas
    custo = 0  # define a intensidade do custo computacional: 0 para baixo, 1 para médio e 2 para alto
    opcao_grafo_grande = 0

    if g.ecount() <= arestas_para_custoso:
        custo = 0
    else:
        custo = 1

    nome_dados = ""
    # se o usuário optou por gerar uma imagem do grafo ou realizar alguma medida
    if not nao_gerar_imagem_grafo or lista_medidas != ["none"]:
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
        nome_dados = monta_nome(limiar, lista_atributos_numerico, nome_arquivo_csv)

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

    if nova_lista_medidas != ["none"]:
        print("Generating table...")
        if len(nova_lista_medidas) != 0:
            if g.vcount() != 0:
                nome_tabela = f"table_{nome_dados}.pdf"
                nome_tabela_freq = f"freq_table_{nome_dados}.pdf"
                # tabela com as medidas de caracterização selecionadas é gerada
                monta_tabela(dados=calcula_medidas(g, nova_lista_medidas,
                             g.vs["label"]), nome=nome_tabela, tipo="centrality")
                # tabela de frequências é gerada
                monta_tabela(dados=calculate_frequency_keys(g, "Link"), nome=nome_tabela_freq, tipo="frequency")
            else:
                print("Empty graph, no table generated")
        else:
            print("Centrality measurements list is empty")

    dict_vizinhos = cria_dicionario_vizinhos_links(
        g, keys, intervalo=interval_amplitude, max_step=calcula_max_step(lista_dict, keys))

    # Saves dictionary to pickle file

    dict_path = Path("results/dictionaries")
    dict_path.mkdir(exist_ok=True, parents=True)
    dict_pickle_file_name = Path(f"dict_{gets_name_file(nome_arquivo_csv)}.pkl")
    with open(dict_path/dict_pickle_file_name, "wb") as dict_pickle_file:
        pickle.dump(dict_vizinhos, dict_pickle_file)
    print(f"Generated dict file '{str(dict_pickle_file_name)}' at {str(dict_path)}")

    t_total = time.time() - t_inicio  # Temporizador de saída
    print(f"Finished in {t_total:.4f} seconds\n")


if __name__ == "__main__":
    main()
