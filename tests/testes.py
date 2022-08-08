arestas_para_custoso = 2000 # quantidade de arestas para que o grafo seja considerado custosos

parser = ap.ArgumentParser()
parser.add_argument("-f", "--arquivo", help='''Nome do arquivo csv contendo os dados que será usado para gerar o grafo. Tipo: string. Se o diretório do arquivo não for o mesmo do main.py, indicar o caminho para o arquivo. Exemplo no mesmo diretório do script: -f "meuarquivo.csv". Exemplo em um diretório diferente: -f "/home/user/caminho_para_arquivo/meuarquivo.csv"''')
parser.add_argument("-atb", "--atributos", default=["ALL"], nargs="+", help="Lista de atributos considerados para montar as arestas. Será passado o número da coluna correspondente ao atributo. Este número pode ser visto com o script mostra_colunas.py. Se nenhum atributo for passado, como padrão, serão usados todas as colunas menos as que compõem o id do grafo. Tipo: int. Exemplo: -atb 3 4. Pode também ser passado um intervalo de números no formato inicial-final. Exemplo: -atb 1-3 irá gerar uma lista com 1, 2 e 3.")
parser.add_argument("-id", "--identificadores", nargs="+", help="Lista de atributos que irão compor o label dos vértices do grafo. Pode ser passado apeanas um atributo ou múltiplos, que serão concatenados. Será passado o número da coluna correspondente ao atributo. Este número pode ser visto com o script mostra_colunas.py. Tipo: int. Se o label dos vértices não for único, o programa irá informar sobre e irá encerrar. Exemplo: -id 1 2. Pode também ser passado um intervalo de números no formato inicial-final. Exemplo: -atb 1-3 irá gerar uma lista com 1, 2 e 3.")
parser.add_argument("-rst", "--restricao", default=["none"], nargs="+", help="Lista de atributos usados como restrição para criar uma aresta, isto é, se, entre dois vértices, o valor do atributo restritivo é o mesmo, a aresta não é criada. O padrão é nenhuma restrição. Podem ser informadas múltiplas restrições. Será passado o número da coluna correspondente ao atributo. Este número pode ser visto com o script mostra_colunas.py. Tipo: int. Exemplo: -rst 2. Pode também ser passado um intervalo de números no formato inicial-final. Exemplo: -atb 1-3 irá gerar uma lista com 1, 2 e 3.")
parser.add_argument("-lim", "--limiar", type=float, default=0, help="Limiar usado para gerar as arestas do grafo. O padrão é 0 (zero). Tipo: float. Exemplo: -lim 0.001")
parser.add_argument("-o", "--usar_or", action="store_true", default=False, help="Usa a lógica OR para formar as arestas, isto é, para ser criada uma aresta, pelo menos um dos atributos passados na lista de atributos tem que estar dentro do limiar. O padrão é AND, ou seja, todos os atributos passados têm que estar dentro do limiar.")
parser.add_argument("-m", "--medidas",  default=["none"], nargs="+", help=f'''Lista de medidas de centralidade que serão tomadas sobre o grafo, as quais serão registradas em uma tabela. O padrão é nenhuma, de modo que nenhuma tabela será gerada. Se for muito custoso para tomar a medida (o grafo tem mais de {arestas_para_custoso} arestas), o programa irá perguntar ao usuário se ele realmente quer tomá-la. Podem ser informadas múltiplas medidas. Tipo: string. Exemplo: -m "degree" "betweenness"''')
parser.add_argument("-ni", "--no_graph_image", action="store_true", default=False, help=f"Define se será gerada uma imagem para o grafo. O padrão é gerar uma imagem, se este parâmetro for indicado, não será gerada uma imagem do grafo. Se este tiver mais de {arestas_para_custoso} arestas, será perguntado se realmente quer gerar a imagem, dado o custo computacional da tarefa.")
parser.add_argument("-rgraph", "--raw_graph", action="store_true", default=False, help="Determina se o grafo usado para tomar as medidas inclui vértices que não formam nenhuma aresta. Por padrão, o programa remove os vértices de grau zero do grafo. Com esta opção, o programa usará o grafo sem remover estes vértices.")
parser.add_argument("-giant", "--giant_component", action="store_true", default=False, help="Determina se apenas o giant component é mostrado na imagem ou se todo o grafo é mostrado. O padrão é mostrar todo o grafo.")
parser.add_argument("-rdata", "--raw_data", action="store_true", default=False, help="Determina se o programa normaliza os dados de entrada ou não. O padrão é normalizar, ou seja, não usar os dados puros.")
parser.add_argument("-mdeg", "--min_degree", type=int, default=0, help="Ao mostrar o grafo, apenas serão plotados os vértices com grau a partir do especificado. O padrão é 0, ou seja, sem restrições para degree. Tipo int. Exemplo: -mdeg 1")
parser.add_argument("-mstep", "--min_step", type=int, default=0, help="Step a partir do qual os vértices irão compor o grafo final. Tipo int. O padrão é 0. Ex: -mstep 3000. Isso significa que apenas vértices cujo step é maior ou igual a 3000 irão compor o grafo. Importante: é necessário que, no arquivo de entrada, a coluna referente ao step seja nomeada 'Step' ou 'step' para que o programa possas reconhecê-la.")

args = parser.parse_args()

nome_arquivo_csv = args.arquivo # nome do arquivo csv a ser usado para gerar grafo
lista_atributos_numerico = args.atributos # lista de atributos, passados como número da coluna
lista_ids_label_numerico = args.identificadores # lista de ids usados no label, passados como número da coluna
lista_restricoes_numerico = args.restricao # lista de restricoes para criar arestas, passadas como número da coluna
limiar = args.limiar  # limiar usado para criar arestas
usar_or = args.usar_or  # lógica para criar arestas
lista_medidas = args.medidas  # lista de medidas que serão tomadas do grafo
nao_gerar_imagem_grafo = args.no_graph_image # define se será gerada uma imagem do grafo ou não
usar_grafo_puro = args.raw_graph # define se será usado o grafo sem processamento (remover vértices de grau zero) ou não
giant_component = args.giant_component # define se apenas o giant component será mostrado na imagem
use_raw_data = args.raw_data  # define se os dados usados serão normalizados
min_degree = args.min_degree # apenas serão mostrados vértices com grau a partir do especificado
min_step = args.min_step # apenas serão considerados vértices cujo step é maior ou igual a este valor

# == Verfica consistência de entrada == 

    if args.identificadores == None:
        print("Erro na passagem de parâmetro, o campo dos ids não foi informado.")
        sys.exit("Saindo do programa")

    if args.arquivo == None:
        print("Erro na passagem de parâmetro, o campo do nome do arquivo não foi informado.")
        sys.exit("Saindo do programa")

    print("Parâmetros OK")  # se todos os parâmetros necessário foram informados
