# Simulations for data creation
**Note**: to run the scripts listed here, you will need to download the necessary data from the [OSF repository](https://osf.io/mde4k/) corresponding to this project. Please download the folder ```data``` from the repository and place it in the main directory of this repository (i.e. as ```../../data```, relative to the location of this README file).

Running the simulations with different measure combinations for all shool types is done in three steps:
1. First, the contact networks for all the school types with representative school layouts (average number of students and students per class) need to be created.
2. Once the contact networks exist, simulations are run on these contact networks to screen all possible combinations of measures.
3. After ensembles of simulations have been created for each school type and parameter combination, the ensemble statistics can be calculated and visualised.


## Creation of contact networks
The creation of contact networks is performed in the script ```construct_school_networks_representative_schools.ipynb```. The script does not require any external data to run as all relevant parameters pertaining to the design of contact networks are set in the script. The result of this are the contact networks, node lists and schedules for all school types, saved to the folder ```../../data/contact_networks/representative_schools```. In case you do not want to create your own contact networks, we uploaded contact networks for representative schools as compressed archives in the OSF repository at ```../../data/contact_networks/representative_schools/school_type.zip```. To use these contact networks, please decompress the archives first.

## Running the simulations
Simulations are run for all 288 possible combinations of measures (different preventive screening frequencies, mask wearing, ventilation and class size reductions) and all school types in the script ```intervention_measures_data_creation.ipynb```.  

The simulation parameters, prevention measures and school characteristics used for these simulations are stored separately in the folder ```params```. As base transmission risk, intermediate contact weight, far contact weight and age transmission discout we use the values determined during the calibration of the model.  

The module ```data_creation_functions.py``` provides the main functions used to run the simulations, aggregate simulation results and extract observables. Simulation results are stored in the folder ```../../data/intervention_measures/simulation_results```.

**Note**: Each measure combination and school type requires at least 500 runs to produce reliable statistics. This takes a long time and should therefore be run on several cores in parallel. Here, we provide a sample script to test the functionality of the simulations. We ran and aggregated all simulations on a cluster and provide the resulting ensembles and observables for every school type in the folder ```../../data/intervention_measures/simulation_results```.

## Analysing the results
Analysis of the ensembles of simulations for each school type and parameter combination is done in the script ```intervention_measures_analysis.ipynb```. In this script, the impact of individual measures (figure 3 in the paper) and of measure combinations (figure 4) is analysed. Plots are stored at ```../../plots/intervention_measures```.