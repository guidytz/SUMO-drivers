# SUMO-QL

A python code to handle Multi-agent Reinforcement Learning using [SUMO](https://github.com/eclipse/sumo) as a microscopic 
traffic simulation. 

> Currently working with SUMO v.1.9.2

## Dependencies
SUMO environment needs to be installed in order to run the code correctly.

### SUMO Installation

In Ubuntu, simply run:

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

### Adding traci, sumolib and libsumo as python packages
To be able to run the code, is necessary to install traci, sumolib and libsumo as python packages.
Simply run:
```
pip install traci sumolib libsumo
```

## Usage

```bash
python3 simulations/sumo_ql_run.py -c <scenario>
```

Where the scenario is a basic .sumocfg file containing info about network and route files necessary for the simulation. Some
examples can be found in [scenario](https://github.com/guidytz/SUMO-QL/tree/master/scenario).

This will run the application with the chosen scenario, applying Q-Learning algorithm to each car agent coupled with car
to infrastructure (C2I) communication.

Different parameters can be set in order to the simulation to behave differently. See which ones are available by running 
the command below:

```bash
python3 simulations/sumo_ql_run.py -h
```

After running the experiments, results can be found in results folder generated.

## Documentation
It is possible to see the module documentation using [pdoc](https://pdoc3.github.io/pdoc/). 
Just install pdoc using:
```bash
pip install pdoc
```

Then run the following line to open a server with the documentation:
```bash
pdoc3 --http : sumo_ql
```

## Utilities

Some utility scripts were implemented to be able to plot the results, which can be used as the example below:

```bash
python3 utilities/plot_ma.py -f <path_to_csv_file>
```
