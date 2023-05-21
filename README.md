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

This tool receives data from a traffic simulation and creates a graph that links different elements of the network that have similar patterns. This virtual graph can be used to enhance the exchange of information between CommDevs during the simulation. It is also possible to use it to study the network itself, taking different centrality measures of this graph.

### Communication with Virtual Graph

Creating the virtual graph at the start of the simulation using the virtual graph specific arguments or loading it from a [pickle](https://docs.python.org/3/library/pickle.html) file:

### Creating the Virtual Graph:

I

### Loading from file:

### Taking measurements from the Virtual Graph:

### Utility tools for the Virtual Graph:

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
