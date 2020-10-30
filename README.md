# SUMO-QL

A python code to handle Multiagent Reinforcement Learning using [SUMO](https://github.com/eclipse/sumo) as a microscopic 
traffic simulation. 

> currently working with SUMO v.1.7.0

## Dependencies
SUMO environment need to be installed in order to run the code correctly.

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

Other options for installing SUMO in different systems can be found in [SUMO's documantation page](https://sumo.dlr.de/docs)

## Usage

```bash
python3 main.py -c <scenario>
```

Where the scenario is a basic .sumocfg file containg info about network and route files necessary for the simulation

This will run the application with the chosen scenario, applying Q-Learning algorithm to each car agent coupled with car
to infrastructure (C2I) communication.

Different parameters can be set in order to the simulation to behave differently. See which ones are available by running 
the command below:

```bash
python3 main.py -h
```

After running the experiments, results can be found in [csv](https://github.com/guidytz/SUMO-QL/tree/master/csv).

## Utilities

Some utilities scripts were implemented to be able to plot the results, which can be used as the example below:

```bash
python3 utilities/plot_ma.py -f <path_to_csv_file>
```
