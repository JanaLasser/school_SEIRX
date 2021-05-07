# Simulations for sensitivity analysis
**Note**: to run the scripts listed here, you will need to download the necessary data from the [OSF repository](https://osf.io/mde4k/) corresponding to this project. Please download the folder ```data``` from the repository and place it in the main directory of this repository (i.e. as ```../../data```, relative to the location of this README file).

The sensitivity analysis includes two components:
1. An analysis of a situation with increased transmission risk.
2. An analysis of a "worst case" scenario with badly implemented measures


## Increased transmission risk
Increased transmission risk simulations are run on the same contact networks of representative schools as the initial intervention measure investigation (see README of section "intervention measures"). Contact networks, node lists and schedules for all school types are stored in the folder ```../../data/contact_networks/representative_schools```.

Simulations are run for all 288 possible combinations of measures (different preventive screening frequencies, mask wearing, ventilation and class size reductions) and all school types but with an increased transmission risk. The script ```run_data_creation_transmissibility.py``` takes command line parameters as inputs and runs all required simulation runs for one ensemble. The python-script can be launched by the bash script ```submit_jobs_transmissibility.sh``` to be run on several cores in parallel. Simulation results are stored in the folder ```../../data/sensitivity_analysis/simulation_results/transmissibility/```. Results are aggregated in the script ```sensitivity_analysis_extract_observables.ipynb```.

**Note**: Since running all ensembles with the necessary number of runs to create reliable statistics takes long, we provide the aggregated and extracted ensemble observables in the OSF repository at ```data/sensitivity_analysis/simulation_results/transmissibility/```. 

## Worst case scenarios
Some of the worst case scenarios involve a reduction of class sizes. This requires us to create new contact networks with a newly defined class size reduction ratio. Creating these contact networks is done in the script ```construct_school_networks_reduced_class_sizes.ipynb```. The corresponding contact networks, node lists and schedules are stored in the folder ```../../data/contact_networks/reduced_class_size```. Again, we provide these contact networks as compressed archives in the OSF repository as ```data/contact_networks/reduced_class_size/school_type.zip```. If you want to use these pre-made contact networks, please unzip them first.

After the contact networks have been created, simulations for the worst-case scenarios can be run using the script  ```run_data_creation_worst_case.py```. Similar to the investigation of higher transmissibility described above, the python script can be launched on multiple cores in parallel using the bash script ```submit_jobs_worst_case.sh```. Results of these simulations are also aggregated in the script ```sensitivity_analysis_extract_observables.ipynb```. 

**Note**: Since running all ensembles with the necessary number of runs to create reliable statistics takes long, we provide the aggregated and extracted ensemble observables in the OSF repository at ```data/sensitivity_analysis/simulation_results/worst_case/```.

## Analysing the results
Analysis of the ensembles of simulations for the higher transmissibility scenario and the worst case scenario and a comparison to the baseline scenarios is done in the sript ```sensitivity_analysis.ipynb```. In this script, the increase of outbreak sizes compared to the baseline scenario (figure 5 in the paper) is analysed and visualised. The figure is stored at ```../../plots/sensitivity_analysis```.