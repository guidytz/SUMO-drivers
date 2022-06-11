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

## Basic Usage

```bash
python3 simulations/sumo_run.py -c <scenario>
```

Where the scenario is a basic .sumocfg file containing info about network and route files necessary for the simulation. Some
examples can be found in [scenario](https://github.com/guidytz/SUMO-QL/tree/master/scenario).

This will run the application with the chosen scenario, applying Q-Learning algorithm to each car agent.

## Other Configurations

### Number of Steps

To define the number of steps to run the simulations, use the argument _-s_ followed by the value desired. For example, to run 
the 5x5 network present in the scenario folder for 30.000 steps, run the following:
```bash
python3 simulations/sumo_run.py -c scenario/5x5_allway_stop/5x5.sumcfg -s 30000
```

### Steps Before Learning Starts

In order to populate the network before staring the learning process, it is important to define a number of steps in which there 
is no learning algorithm involved and the agents just follow the routes present in the SUMO configuration file. This setting is
specified with the _-w_ argument. So, for exemple, if we want to run the 5x5 network for 30.000 steps and populate the network
for 4.000 steps, run the following:
```bash
python3 simulations/sumo_run.py -c scenario/5x5_allway_stop/5x5.sumcfg -s 30000 -w 4000
```

This also allows us to run the simulation without using a learning algorithm by simply using the same value for the steps and
populating steps. For example, running the 5x5 network for 30.000 steps without the agents learning:
```bash
python3 simulations/sumo_run.py -c scenario/5x5_allway_stop/5x5.sumcfg -s 30000 -w 30000
```

### Communication Success Rate

This approach allows the learning process to be improved with the use of [Car-to-Infrastructure Communication](https://peerj.com/articles/cs-428/), this can
be set by adjusting the success rate of the communication using the _-r_ argument. So, if we want to run a simulation that has 
a communication with a 75% success rate, we follow the example below:
```bash
python3 simulations/sumo_run.py -c scenario/5x5_allway_stop/5x5.sumcfg -s 30000 -r 0.75
```

At the current version, the only algorithm that supports communication is the base Q-Learning algorithm, so the communication
will not work with multiobjective scenarios.

### Multiobjective Learning

To be able to test multiobjective learning, we need to prepare some settings. First, it is important to run normal Q-Learning
with each objective separately in order to collect important data to normalize the multiobjective run. For example, if we want
to optimize travel time and carbon monoxide emission in a multiobjective run, we first run the collect runs for each of them like
the following:
```bash
python3 simulations/sumo_run.py -c scenario/5x5_allway_stop/5x5.sumcfg -a QL --objectives TravelTime CO
```

```bash
python3 simulations/sumo_run.py -c scenario/5x5_allway_stop/5x5.sumcfg -a QL --objectives CO TravelTime
```

The _-a_ argument sets the Q-Learning algorithm as the main learning algorithm the agents will use, the _--objectives_ argument
sets the objectives to collect information of. Notice that the Q-Learning algorithm will only optimize the first objective stated
in the informed list.

After collecting has been done, it is possible to use the multiobjective algorithm that aims at optimizing all objetives informed.
Following the example aforementioned, to optimize travel time and carbon monoxide using Pareto Q-Learning, run the following:
```bash
python3 simulations/sumo_run.py -c scenario/5x5_allway_stop/5x5.sumcfg -a PQL --objectives TravelTime CO
```

### Additional Arguments

Different arguments can be set in order to the simulation to behave differently. See which ones are available by running 
the command below:

```bash
python3 simulations/sumo_run.py -h
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
