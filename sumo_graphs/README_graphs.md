Henrique Uhlmann Gobbi - UFRGS
# Comunicação e Aprendizado de Máquina em Mobilidade Urbana: uma Abordagem Multiagente e Multiobjetivo

# Instruções para rodar script para gerar um grafo

# Dependências:
	- python3
	- igraph (https://igraph.org/python/)
	- argparse (https://pypi.org/project/argparse/)
	- matplotlib (https://matplotlib.org/stable/users/installing/index.html)
	- pandas (https://pypi.org/project/pandas/)
	
# Para executar:
	- colocar os scripts em um mesmo diretório
	- o principal (main) é o main.py, os outros arquivos contém funções para cada etapa do processo de geração do grafo
	- comandos:	
		> $ python3 main.py -h
			output: explica os argumentos que devem ser passados
		> $ python3 main.py [argumentos]
			output: informações no terminal, arquivo texto com dados do grafo, tabela com algumas medidas de caracterização e uma imagem do grafo
			
# Exemplos de uso:
    O usuário pode gerar um grafo a partir de um csv, indicando os atributos que unem os vértices, atributos restritivos e um limiar. A partir desse grafo, pode ser gerada uma imagem dele e/ou uma tabela com medidas de caracterização passadas pelo usuário. Os valores dos atributos são normalizados por padrão para facilitar a escolha de valor para o limiar, mas pode ser usado os valores originais com o comando -rdata.

    1) Usando csv inteiro
    	Se o usuário quiser usar seu dataset completo, estes são os passos para uma execução normal:
    	
    	i) Primeiro, o usuário pode mostrar os índices das colunas do seu arquivo csv, que vão ser usadas para indicar os atributos usados:
    		> $ python3 mostra_colunas.py -f meudataset.csv
    		output: 
    			1 -> Atributo 1
    			2 -> Atributo 2
    			3 -> Atributo 3
    			4 -> Atributo 4
    			...
    			n -> Atributo n
    	
    	ii) Com a informação dos ids, o usuário pode, então usar o script principal para gerar o grafo:
    		> $ python3 main.py -f meudataset.csv -atb 1 2 -id 3 -rst 4 -lim 0.1
    		output: 
    			Será gerado um grafo a partir do arquivo indicado, usando os atributos 1 e 2 para unir os vértices, 3 como id do grafo (label), 4 como atributo 		restritivo e limiar de 0.1. Há outras opções que o usuário pode escolher, como está indicado no -h. Dentre elas, por exemplo, temos que, com -m o usuário 			informa as medidas de centralidade que quer tomar, com -ni especifica que não quer gerar uma imagem do grafo ou com -o indica se quer usar a lógica or 			para montar as arestas.
    			Assim, como o usuário não usou -ni, será produzida uma imagem do grafo, e como não foi passado nenhum -m, não será gerada nenhuma tabela.


    2) Usando csv reduzido
    	Se o usuário quer produzir um dataset menor a partir do seu completo:
    	
    	O usuário tem a opção de escolher entre especificar o número de linhas do csv reduzido ou, com o argumento -p, quantos por cento do csv original ele quer usar para gerar o csv menor.
    	
    	i) Primeiro, gera-se o csv menor:
    		> $ python3 gera_csv_menor -f meusdados.csv -n 10
    		output:
    			Será gerado um arquivo csv com 10 linhas aleatórias do csv original (se fosse utilizado -p, seria gerado um arquivo com 10% do número de linhas do original)
    	
    	ii) Em seguida, mostra-se os ids das colunas que podem ser usadas para indicar os atributos usados:
    		> $ python3 mostra_colunas -f meusdados_com_10_linhas.csv
    		output:
    			1 -> Atributo 1
    			2 -> Atributo 2
    			3 -> Atributo 3
    			4 -> Atributo 4
    			...
    			n -> Atributo n
    	
    	iii) Finalmente, o usuário pode rodar o script principal:
    		> $ python3 main.py -f meusdados_com_10_linhas.csv -atb 1 2 -id 4 -rst 3 -lim 0.1 -m degree
    		Será gerado um grafo a partir do arquivo reduzido, com os atributos 1 e 2 formando arestas, o 4 como id e o 3 como restritivo. Além disso, o limiar será 		0.1 e, ao final da execução, será gerada uma tabela com os valores do degree de cada vértice do grafo.
    
# Notas:
	- Se a lista de ids gerada não for única, o programa irá encerrar.
	- Se nenhum atributo for informado em -atb, o programa usará todas as colunas do csv menos as que compõem o id.
	- É necessário informar uma lista de ids e o arquivo utilizado, senão o programa encerra.
	- Se o grafo estiver vazio, não será gerada nenhuma imagem e nenhuma tabela de medidas.
	- A utilização de aspas duplas ("") não é estritamente necessária, se o nome passado não tiver espaços, podendo também ser utilizadas aspas simples (''). 
    
# Execuções:
    
    1)  > $ python3 main.py -atb 4 5 -id 1 2 -rst 7 -lim 0.1 -f meu_arquivo.csv -giant
	
	Atributos usados: "Meu Atributo 1 Norm"
	Label dos vértices será formado por: "Meu Atributo 1" "Meu Atributo 2"
	Atributos restritivos: "Meu Atributo 2 Norm"
	Limiar: 0.1
	Lógica: AND
	Arquivo usado: "meu_arquivo.csv"
	Medidas calculadas: não gera tabela
	Gera imagem do grafo: True
	Usar grafo puro: False
	Mostrar apenas giant component: True
	Mostra vértices com degree a partir de: 0
	
    2)  > $ python3 main.py -atb 4 5 -id 1 2 -rst 7 -lim 0.1 -f meu_arquivo.csv -o -ni -rgraph -m degree eigenvector_centrality -mdeg 2
	
	Atributos usados: "Meu Atributo 1 Norm"
	Label dos vértices será formado por: "Meu Atributo 1" "Meu Atributo 2"
	Atributos restritivos: "Meu Atributo 2 Norm"
	Limiar: 0.1
	Lógica: OR
	Arquivo usado: "meu_arquivo.csv"
	Medidas calculadas: "degree" "eigenvector_centrality"
	Gera imagem do grafo: False
	Usar grafo puro: True
	Mostrar apenas giant component: False
	Mostra vértices com degree a partir de: 2
	
# Scripts adicionais:
    *)  mostra_colunas.py
		- função: dado um arquivo csv, mostra as colunas deste arquivo com seus respectivos ids, que podem ser usadas como atributos para formar o grafo
		- comandos:  
			> $ python3 mostra_colunas.py -h
			> $ python3 mostra_colunas.py -f [nome do arquivo]
		- exemplo:
			> $ python3 mostra_colunas.py -f meu_arquivo.csv
			
    *)  gera_csv_menor.py
		- função: dado um arquivo csv, que será a fonte, e um número, que pode ser o número de linhas retirados da fonte ou a porcentagem de linhas extraídas do csv original. A função gera um arquivo csv com linhas aleatórias do arquivo csv fonte; a quantidade de linhas é dada pelo número passado como argumento
		- comandos:  
			> $ python3 gera_csv_menor.py -h
			> $ python3 gera_csv_menor.py -f [nome do arquivo] -n [número de linhas] -p [se indicado, o número passado representará uma porcentagem]
		- exemplo:
			> $ python3 gera_csv_menor.py -f meu_arquivo.csv -n 10
			
# Sobre os parâmetros:

    -f : Nome do arquivo csv contendo os dados que será usado para gerar o grafo. Tipo: string. Se o diretório do arquivo não for o mesmo do main.py, indicar o caminho para o arquivo. Exemplo no mesmo diretório do script: -f meuarquivo.csv. Exemplo em um diretório diferente: -f "/home/user/caminho_para_arquivo/meuarquivo.csv
    
    -atb : Lista de atributos considerados para montar as arestas. Será passado o número da coluna correspondente ao atributo. Este número pode ser visto com o script mostra_colunas.py. Se nenhum atributo for passado, como padrão, serão usados todas as colunas menos as que compõem o id do grafo. Tipo: int. Exemplo: -atb 3 4. Pode também ser passado um intervalo de números no formato inicial-final. Exemplo: -atb 1-3 irá gerar uma lista com 1, 2 e 3.
    
    -id : Lista de atributos que irão compor o label dos vértices do grafo. Pode ser passado apeanas um atributo ou múltiplos, que serão concatenados. Será passado o número da coluna correspondente ao atributo. Este número pode ser visto com o script mostra_colunas.py. Tipo: int. Se o label dos vértices não for único, o programa irá informar sobre e irá encerrar. Exemplo: -id 1 2. Pode também ser passado um intervalo de números no formato inicial-final. Exemplo: -atb 1-3 irá gerar uma lista com 1, 2 e 3.
    
    -rst : Lista de atributos usados como restrição para criar uma aresta, isto é, se, entre dois vértices, o valor do atributo restritivo é o mesmo, a aresta não é criada. O padrão é nenhuma restrição. Podem ser informadas múltiplas restrições. Será passado o número da coluna correspondente ao atributo. Este número pode ser visto com o script mostra_colunas.py. Tipo: int. Exemplo: -rst 2. Pode também ser passado um intervalo de números no formato inicial-final. Exemplo: -atb 1-3 irá gerar uma lista com 1, 2 e 3.
    
    -lim : Limiar usado para gerar as arestas do grafo. O padrão é 0 (zero). Tipo: float. Exemplo: -lim 0.001
    
    -o : Usa a lógica OR para formar as arestas, isto é, para ser criada uma aresta, pelo menos um dos atributos passados na lista de atributos tem que estar dentro do limiar. O padrão é AND, ou seja, todos os atributos passados têm que estar dentro do limiar.
    
    -m : Lista de medidas de centralidade que serão tomadas sobre o grafo, as quais serão registradas em uma tabela. O padrão é nenhuma, de modo que nenhuma tabela será gerada. Se for muito custoso para tomar a medida, o programa irá perguntar ao usuário se ele realmente quer tomá-la. Podem ser informadas múltiplas medidas. Tipo: string. Exemplo: -m degree betweenness
    
    -ni : Define se será gerada uma imagem para o grafo. O padrão é gerar uma imagem, se este parâmetro for indicado, não será gerada uma imagem do grafo.
    
    -rgraph : Determina se o grafo usado para tomar as medidas inclui vértices que não formam nenhuma aresta. Por padrão, o programa remove os vértices de grau zero do grafo. Com esta opção, o programa usará o grafo sem remover estes vértices.
    
    -giant : Determina se apenas o giant component é mostrado na imagem ou se todo o grafo é mostrado. O padrão é mostrar todo o grafo.
    
    -rdata : Determina se o programa usa os dados de entrada sem normalizar. O padrão é normalizar.
    
    -mdeg : Ao mostrar o grafo, apenas serão plotados os vértices com grau a partir do especificado. O padrão é 0, ou seja, sem restrições para degree. Tipo int. Exemplo: -mdeg 2
    
    -mstep : Step a partir do qual os vértices irão compor o grafo final. Tipo int. O padrão é 0. Exemplo: -mstep 3000. Isso significa que apenas vértices cujo step é maior ou igual a 3000 irão compor o grafo. Importante: é necessário que, no arquivo de entrada, a coluna referente ao step seja nomeada 'Step' ou 'step' para que o programa possas reconhecê-la.
    

# Medidas de centralidade:

    A sintaxe das principais medidas de centralidade que podem ser tomadas pelo parâmetro -m é a seguinte:
    
    Degree:	      "degree"
    Betwenness:       "betweenness"
    Closeness: 	      "closeness"
    Eigenvector:      "eigenvector_centrality"
    Pagerank: 	      "pagerank"
    Diversity: 	      "diversity"
    Constraint:       "constraint"
    Eccentricity:     "eccentricity"
    Strength: 	      "strength"
    
    Mais medidadas podem ser encontradas em: https://igraph.org/python/tutorial/latest/analysis.html#vertex-properties
    Para usá-las, bastar passar, no -m, o nome do atributo como aparece na página indicada.

