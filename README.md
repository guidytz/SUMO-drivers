# SUMO-QL

A python code to handle Multi-agent Reinforcement Learning using [SUMO](https://github.com/eclipse/sumo) as a microscopic
traffic simulation.

## Requirements

- Python 3.10+.
- SUMO v 1.8.0.

### SUMO Installation

In Ubuntu, run:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

After installing SUMO, it's necessary to setup SUMO_HOME as an environment variable, which is done by adding the
following line to .bashrc:

```bash
export SUMO_HOME="/usr/share/sumo"
```

Other options for installing SUMO in different systems can be found in [SUMO's documentation page](https://sumo.dlr.de/docs)

### Installing the package

To install the package, run:

```bash
git clone https://github.com/guidytz/SUMO-QL
cd SUMO-QL
python3 -m pip install -e .
```

## Usage

The script has three basic agent modes:

- Non-Learning Agent
- Q-Learning Agent
- Pareto Q-Learning Agent

Each of these agents have their specific parameters that can be passed through command line, but all of them require a .sumofg file
which contains configurations involving the SUMO network in use.

Examples of basic usage with each agent are given below:

### Non-Learning Agent

Using the positional argument `nl`:

```
python3 simulations/sumo_run.py nl --sumocfg <path-to-sumocfg-file>
```

### Q-Learning Agent

Using the positional argument `ql`:

```
python3 simulations/sumo_run.py ql --sumocfg <path-to-sumocfg-file>
```

### Pareto Q-Learning Agent

Using the positional argument `pql`:

```
python3 simulations/sumo_run.py pql --sumocfg <path-to-sumocfg-file>
```

## Virtual Graph

This tool receives data from a traffic simulation and creates a graph that links different elements of the network that have similar 
patterns. This virtual graph can be used to enhance the exchange of information between CommDevs during the simulation. It is also 
possible to use it to study the network itself, taking different centrality measures of this graph.

### Creating the Virtual Graph

The first step is to generate the input file of the virtual graph. This can be done by running a simulation using the **Non-Learning Agent**:

```
python3 simulations/sumo_run.py nl --sumocfg <path-to-sumocfg-file> --observe-list <attributes-to-gather-data-from>
```

The `observe-list` argument contains the names of the attributes of the simulation that will be present in the output csv file.
This will generate two csv files with information about the network in the results folder. One contains data of each link at a timestep interval and the other
contains this same data aggregated by traffic light junction at each timestep interval. The next step is to run the virtual graph script using the 
[virtual graph specific arguments](#virtual-graph-specific-arguments):

```
python3 sumo_vg/run_virtual_graph --vg-file <path-to-vg-input-file> --vg-attributes <list-of-attributes> --vg-label <list-of-labels> --vg_threshold <threshold-of-the-virtual-graph>
```

The program will compare every line of the input csv with every other line, checking if the difference between the chosen attributes is within a defined threshold. Every attribute is normalized between $0$ and $1$ using the formula:
```math
x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min}}
```

Every argument that refers to the attributes in the input csv file, such as the list of attributes or the list of labels, is passed using its
respective column number in the input csv. They can be passed as a list of integers or as an interval in the form `initial_column-final_column` 
where `initial_column` is the first column of the interval and `final_column` is the last.

### Output of the Virtual Graph

The script will generate an image of the virtual graph in the results/graphs folder. Most importantly, it will also generate a [pickle](https://docs.python.org/3/library/pickle.html) 
file containing a python dictionary with every link or junction and their respective neighbors in the virtual graph at each timestep interval.

### Using Link or Junction as Graph Vertex

The first column passed as label for the virtual graph with the `vg-label` argument will be used as the vertex attribute of the virtual graph. This determines if the program 
will aggregate the virtual graph neighbors by link or by junction in the output dictionary file.

### Communication with Virtual Graph

[Creating the virtual graph](#creating-the-virtual-graph) at the start of the simulation using the [virtual graph specific arguments](#virtual-graph-specific-arguments) or [loading it](#loading-from-file) from a [pickle](https://docs.python.org/3/library/pickle.html) file:

### Loading from File

Using the `vg-dict-file` argument:

```
python3 simulations/sumo_run.py ql --sumocfg <path-to-sumocfg-file> --vg-dict-file <path-to-vg-dict-file>
```

With this argument, the communication between CommDevs in the network will be enhanced by the information in the virtual graph dictionary file. It is important to note
that this works with q-learning agents as well as with pareto q-learning agents. Also, it is necessary to set a value different than zero for the `success-rate` argument in 
order to have a working communication during the simulation.

### Example of Use

A simple example of the pipeline described is the following:

1. Generate the input csv file:
```
python3 simulations/sumo_run.py nl --sumocfg scenario/diamond/diamond.sumocfg --observe-list TravelTime CO
```
This will generate the two csv files: one with information about each link and another with information about each traffic light junction. Only one of these files will be
used in the next step.

2. Generate the virtual graph dictionary file:
```
python3 sumo_vg/run_virtual_graph.py --vg-file <path-to-vg-input-file> --vg-attributes 3 4 --vg-label 2 1 --vg_threshold 0.0001 --vg-restrictions 2
```
In the `vg-file` argument, it is chosen whether to use the link data or the junction data generated in the previous step. The `vg-attributes` argument determines that two
attributes from the input csv will be used: the third and fourth column attributes. The `vg-label` argument determines that the label for each vertex of the virtual graph
will be composed by the second and first columns in the input csv, and also that the output dictionary file will be aggregated by the second column. The `vg-threshold`
determines the maximum absolute difference each attribute, defined by `vg-attributes`, of each vertex in the virtual graph can have. Finally, the `vg-restrictions` argument set
as the graph vertex attribute is important because it prevents the program from creating an edge between two vertices that have the same graph vertex attribute, i.e. two 
vertices that originate from the same link or junction. This is a good practice, so as to the output dictionary file doesn't have a link or junction as neighbor of itself.

3. Run the simulation with ql-learning agent and the virtual graph enhanced communication:
```
python3 simulations/sumo_run.py ql --sumocfg scenario/diamond/diamond.sumocfg --observe-list TravelTime CO --success-rate 1 -o TravelTime --vg-dict-file <path_to_dictionary_file>
```
This will run a simulation using the virtual graph to enhance the communication, generating two csv files, one with link information and another with traffic light junction
information of the simulation. 

It is also possible to skip the second step and generate the virtual graph at the start of the simulation using the [virtual graph specific arguments](#virtual-graph-specific-arguments) alongside the simulation arguments. This will also run a simulation using the virtual graph to ehnace the communication.

### Taking Measurements from the Virtual Graph

While [creating the virtual graph](#creating-the-virtual-graph), using the `centrality-measures` specific virtual graph argument:

```
python3 sumo_vg/run_virtual_graph --vg-file <path-to-vg-input-file> --vg-attributes <list-of-attributes> --vg-label <list-of-labels> --vg_threshold <threshold-of-the-virtual-graph> --centrality-measures <list-of-centrality-measures>
```

This command will generate the usual virtual graph dictionary file and its image and also two more pdf files: a list of every vertex of the graph and its centrality measures and
also a list of every graph vertex attribute and how many times it appears in the virtual graph, i.e. the frequency of each specific link or junction in the virtual graph. 

Here's a table with the most commons centrality measures that can be taken and their respective keyword argument:

| Name            | Keyword Argument         |
|-----------------|--------------------------|
| Betwenness      | `betweenness`            |
| Closeness       | `closeness`              |
| Constraint      | `constraint`             |
| Degree          | `degree`                 |
| Diversity       | `diversity`              |
| Eigenvector     | `eigenvector_centrality` |
| Eccentricity    | `eccentricity`           |
| Pagerank        | `pagerank`               |
| Strength        | `strength`               |

Other centrality measures can be found [here](https://python.igraph.org/en/stable/analysis.html#vertex-properties). It is necessary to pass the name of the method only.

### Utility Tools for the Virtual Graph

### Common Arguments

Below are described common arguments to every agent.

| Name           | Argument                       | Description                                                                                                                                                                    |
| -------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Steps          | `-s <INT>`<br>`--steps <INT>`  | Number of SUMO steps to run the simulation.                                                                                                                                    |
| Demand         | `-d <INT>`<br>`--demand <INT>` | Desired network demand.                                                                                                                                                        |
| Average Window | `--aw <INT>`                   | Window size to average collected data.                                                                                                                                         |
| GUI Usage      | `--gui`                        | Flag that indicates SUMO GUI usage.                                                                                                                                            |
| Number of Runs | `-n <INT>`<br>`--nruns <INT>`  | Number of multiple simulation repeated runs.                                                                                                                                   |
| Parallel Runs  | `--parallel`                   | Flag that indicates if multiple runs should run in parallel                                                                                                                    |
| Observe List   | `--observe-list <LIST-OF-STR>` | List that indicate parameters to collect observe in data collection.<br>The possible parameters to use in list are described [here](#list-of-possible-observation-parameters). |

## Common Learning Agent Arguments

Below are described common arguments to every learning agent.

| Name                  | Argument                        | Description                                                                                                                                     |
| --------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Wait to Learn         | `--wait-learn <INT>`            | Number of steps to wait until the learning starts                                                                                               |
| Right Arrival Bonus   | `-b <INT>`<br>`--bonus <INT>`   | Bonus to add in the agent's reward if it arrives at its right destination.                                                                      |
| Wrong Arrival Penalty | `-p <INT>`<br>`--penalty <INT>` | Penalty to subtract in the agent's reward if it arrives at a wrong destination.                                                                 |
| Normalize Rewards     | `--normalize-rewards`           | Flag that indicates if rewards should be normalized. <br>Note that this argument requires a previous run with rewards collected.                |
| Collect Rewards       | `--collect-rewards`             | Flag that indicates if rewards should be collected in a collection file.<br>This file is necessary to run a simulation with normalized rewards. |
| Toll Speed            | `--toll-speed <FLOAT>`          | Speed limit in links where the environment should impose a toll on emission.                                                                    |
| Toll Value            | `--toll-value <INT>`            | Toll value to impose on emission.                                                                                                               |

## Q-Learning Agent Specific Arguments

Below are described arguments specific to Q-Learning agent.

| Name      | Argument                          | Description                                                                                                                                      |
| --------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Alpha     | `--alpha <FLOAT>`                 | Agent's learning rate.                                                                                                                           |
| Gamma     | `--gamma <FLOAT>`                 | Agent's discount factor for future actions.                                                                                                      |
| Objective | `-o <STR>`<br>`--objective <STR>` | Agent's main objective to optimize. <br> List of possible objective to optimize are described [here](#list-of-possible-optimization-parameters). |

## Pareto Q-Learning Specific Arguments

Below are described arguments specific to Pareto Q-Learning agent.

| Name       | Argument                                           | Description                                                                                                                                      |
| ---------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Gamma      | `--gamma <FLOAT>`                                  | Agent's discount factor for future actions.                                                                                                      |
| Objectives | `-o <LIST-OF-STR>`<br>`--objectives <LIST-OF-STR>` | Agent's main objectives to optimize.<br> List of possible objective to optimize are described [here](#list-of-possible-optimization-parameters). |

## Communication Specific Arguments

Below are described arguments specific to Car-to-Infrastructure communication (C2I) usage.

| Name         | Argument                 | Description                                                                                             |
| ------------ | ------------------------ | ------------------------------------------------------------------------------------------------------- |
| Success Rate | `--success-rate <FLOAT>` | Value between 0 and 1 indicating the rate of success in which cars communicate with the infrastructure. |
| Queue Size   | `--queue-size <INT>`     | Queue size in which the infrastructure stores rewards collected from agents.                            |

## Virtual Graph Specific Arguments

Below are described arguments specific to virtual graph in communication usage.

| Name                 | Argument                              | Description                                                                                                                                                           |
| -------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Virtual Graph File   | `--vg_file <STR>`                     | Path and name to the file containing the data that is going to be used to create the virtual graph.                                                                   |
| Attributes           | `--vg_attributes <LIST-OF-STR>`       | List of attributes used to create the virtual graph.<br>Attribute is given by the number of the column of the input file.                                             |
| Labels               | `--vg_labels <LIST-OF-STR>`           | List of attributes that will compose the label of the virtual graph. <br>Attribute is given by the number of the column of the input file.                            |
| Restriction          | `--vg_restriction <LIST-OF-STR>`      | List of attributes that the nodes cannot share in order to create an edge in the virtual graph. <br>Attribute is given by the number of the column of the input file. |
| Threshold            | `--vg_threshold <FLOAT>`              | Threshold used to create an edge in the virtual graph.                                                                                                                |
| Use OR logic         | `--use_or_logic`                      | Flag that indicates or logic instead of the and logic to create an edge between nodes given multiple attributes.                                                      |
| Centrality Measures  | `--centrality_measures <LIST-OF-STR>` | List of centrality measures to be taken of the virtual graph.                                                                                                         |
| No Image Flag        | `--no_image`                          | Flag to indicate to the script not to generate a graph image.                                                                                                         |
| Raw Graph Flag       | `--raw_graph`                         | Flag to indicate not to remove nodes with degree zero (i.e. raw graph).                                                                                               |
| Giant Component Flag | `--giant`                             | Flag to indicate that only the giant component of the graph should be presented in its image.                                                                         |
| Normalize            | `--vg_normalize`                      | Flag to indicate if the input data to graph generation should be normalized.                                                                                          |
| Minimum Degree       | `--min_degree <INT>`                  | Determines the minimum degree a node should have in order to be plotted.                                                                                              |
| Maximum Degree       | `--max_degree <INT>`                  | Determines the maximum degree a node should have in order to be plotted.                                                                                              |
| Minimum Step         | `--vg_min_step <INT>`                 | Determines the maximum step a node should have in order to be plotted.                                                                                                |
| Graph Dictionary     | `--vg_dict_file <STR>`                | Path to file containing the python dictionary of the graph.                                                                                                           |
| Interval             | `--interval <INT>`                    | Timestep interval of the neighbors dictionary.                                                                                                                        |

## List of Possible Observation Parameters

| Name                          | Argument Name        |
| ----------------------------- | -------------------- |
| Link Travel Time              | `TravelTime`         |
| Link Halting Vehicles         | `'Halting Vehicles'` |
| Link Carbon Monoxide Emission | `CO`                 |
| Link Carbon Dioxide Emission  | `CO2`                |
| Link Hidrocarbonets Emission  | `HC`                 |
| Link NOx Emission             | `NOx`                |

## List of Possible Optimization Parameters

| Name                             | Argument Name |
| -------------------------------- | ------------- |
| Agent's Travel Time              | `TravelTime`  |
| Agent's Carbon Monoxide Emission | `CO`          |
| Agent's Carbon Dioxide Emission  | `CO2`         |
| Agent's Hidrocarbonets Emission  | `HC`          |
| Agent's NOx Emission             | `NOx`         |
| Agent's Fuel Consumption         | `Fuel`        |

### Performance boost using Libsumo

To increase performance, declare the following environment variable before running the simulation:

```bash
export LIBSUMO_AS_TRACI=1
```

This allows the simulation use Libsumo instead of Traci, which enhances the performance considerably. However, simulations using sumo-gui are not available using this method. See [Libsumo documentation](https://sumo.dlr.de/docs/Libsumo.html).

### Default values

Default values for each argument can be seen by using the scripts help for each agent. As seen in the examples bellow:

```
python3 simulations/sumo_run.py nl -h
```

```
python3 simulations/sumo_run.py ql -h
```

```
python3 simulations/sumo_run.py ql -h
```

## Documentation

It is possible to see the module documentation using [pdoc](https://pdoc3.github.io/pdoc/).
Just install pdoc using:

```bash
python3 -m pip install pdoc
```

Then run the following line to open a server with the documentation:

```bash
pdoc --http : sumo_ql
```

## Utilities

Some utility scripts were implemented to be able to plot the results, which can be used as the example below:

```bash
python3 utilities/plot_ma.py -f <path_to_csv_file>
```
