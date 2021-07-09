# SUMO-QL

A python code to handle Multi-agent Reinforcement Learning using [SUMO](https://github.com/eclipse/sumo) as a microscopic 
traffic simulation. 

## Requirements

* Python 3.8+.
* SUMO v 1.9.2.

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

### Performance boost using Libsumo
To increase performance, declare the following environment variable before running the simulation:
```bash
export LIBSUMO_AS_TRACI=1
```
This allows the simulation use Libsumo instead of Traci, which enhances the performance considerably. However, simulations using sumo-gui are not available using this method. See [Libsumo documentation](https://sumo.dlr.de/docs/Libsumo.html).

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
