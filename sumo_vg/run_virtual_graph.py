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

    vg_args_creation = parser.add_argument_group('Creating Virtual Graph')
    vg_args_adjusting = parser.add_argument_group('Adjusting Virtual Graph')
    vg_args_plotting = parser.add_argument_group('Plotting Virtual Graph')
    vg_args_measuring = parser.add_argument_group('Measuring Virtual Graph')

    vg_args_creation.add_argument("-f", "--vg-file", dest="vg_file",
                        help="Path to csv file that will be used as input for the virtual graph.")

    vg_args_creation.add_argument("-atb", "--vg-attributes", dest="vg_attributes", default=["ALL"], nargs="+",
                        help="List of attributes used to create the virtual graph. Attribute is given by the number of the column "
                        "of the input csv. (default = ['ALL'])")

    vg_args_creation.add_argument("-id", "--vg-label", dest="vg_label", nargs="+",
                        help="List of attributes that will compose the label of each vertex in the virtual graph. Attribute is given "
                        "by the number of the column of the input csv. The first attribute passed will determine which attribute is used "
                        "to aggregate the virtual graph neighbors, i.e. aggregate by link or junction.")

    vg_args_creation.add_argument("-rst", "--vg-restrictions", dest="vg_restrictions", default=None, nargs="+",
                        help="List of attributes that the vertices cannot share in order to create an edge in the virtual graph. Attribute"
                        " is given by the number of the column of the input csv. (default = None)")
    
    vg_args_creation.add_argument("-tsh", "--vg-threshold", dest="vg_threshold", type=float, default=0,
                        help="Threshold used to create an edge in the virtual graph. (default = 0)")
    
    vg_args_adjusting.add_argument("-or", "--use-or-logic", dest="use_or_logic", action="store_true", default=False,
                        help="Flag that indicates or logic instead of the and logic to create an edge between vertices given multiple "
                        "attributes. (default = false)")
    
    vg_args_measuring.add_argument("-ms", "--centrality-measures", dest="centrality_measures", default=None, nargs="+",
                        help="List of centrality measures to be taken of the virtual graph. (default = None)")
    
    vg_args_plotting.add_argument("-ni", "--no-image", dest="no_image", action="store_true", default=False,
                        help="Flag to indicate to the script not to generate the virtual graph image. (default = false)")
    
    vg_args_adjusting.add_argument("-rgraph", "--raw-graph", dest="raw_graph", action="store_true", default=False,
                        help="Flag to indicate not to remove vertices with degree zero. (default = false)")
    
    vg_args_plotting.add_argument("-giant", "--giant-component", dest="giant_component", action="store_true", default=False,
                        help="Flag to indicate that only the giant component of the virtual graph should be presented in "
                        "its image. (default = false)")
    
    vg_args_adjusting.add_argument("-not-norm", "--vg-not-normalize", dest="vg_not_normalize", action="store_true", default=False,
                        help="Flag to indicate to the script not to normalize the input csv data to generate the virtual"
                        " graph. (default = false)")
    
    vg_args_plotting.add_argument("-mdeg", "--min-degree", dest="min_degree", type=int, default=0,
                        help="Determines the minimum degree a vertex should have in order to be plotted in the virtual graph "
                        "image. (default = 0)")
    
    vg_args_plotting.add_argument("-mstep", "--vg-min-step", dest="vg_min_step", type=int, default=0,
                        help="Determines the minimum step a vertex should have in order to be plotted in the virtual graph image. "
                        "(default = 0)")
    
    vg_args_adjusting.add_argument("-int", "--interval", type=int, default=250,
                        help="Amplitude of the timestep interval of the virtual graph neighbors dictionary. (default = 250)")
                        
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
    not_normalize = args.vg_not_normalize  # define se os dados usados serão normalizados
    min_degree = args.min_degree  # apenas serão mostrados vértices com grau a partir do especificado
    min_step = args.vg_min_step  # apenas serão considerados vértices cujo step é maior ou igual a este valor
    interval_amplitude = args.interval  # amplitude of the virtual graph neighbors dictionary

    # == Verfica consistência de entrada ==

    if args.vg_label is None:
        print("Error! Labels parameter wasn't informed!")
        sys.exit("Exiting program")

    if args.vg_file is None:
        print("Error! Input file path and name wasn't informed")
        sys.exit("Exiting program")

    print("Parameters OK")  # se todos os parâmetros necessário foram informados

    vg_neighbors_dict = generate_graph_neighbors_dict(nome_arquivo_csv=nome_arquivo_csv,
                                                                   lista_atributos_numerico=lista_atributos_numerico,
                                                                   lista_ids_label_numerico=lista_ids_label_numerico,
                                                                   lista_restricoes_numerico=lista_restricoes_numerico,
                                                                   limiar=limiar,
                                                                   usar_or=usar_or,
                                                                   lista_medidas=lista_medidas,
                                                                   nao_gerar_imagem_grafo=nao_gerar_imagem_grafo,
                                                                   usar_grafo_puro=usar_grafo_puro,
                                                                   giant_component=giant_component,
                                                                   not_normalize=not_normalize,
                                                                   min_degree=min_degree,
                                                                   min_step=min_step,
                                                                   arestas_para_custoso=2000,
                                                                   precisao=10,
                                                                   intervalo_vizinhos=interval_amplitude,
                                                                   network_name=gets_name_file(nome_arquivo_csv))

    # Saves dictionary to pickle file
    dict_path = Path("results/dictionaries")
    dict_path.mkdir(exist_ok=True, parents=True)
    dict_pickle_file_name = Path(f"dict_{gets_name_file(nome_arquivo_csv)}.pkl")
    with open(dict_path/dict_pickle_file_name, "wb") as dict_pickle_file:
        pickle.dump(vg_neighbors_dict, dict_pickle_file)
    print(f"Generated dict file '{str(dict_pickle_file_name)}' at {str(dict_path)}")

    t_total = time.time() - t_inicio  # Temporizador de saída
    print(f"Finished in {t_total:.4f} seconds\n")

if __name__ == "__main__":
    main()
