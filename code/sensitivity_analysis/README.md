# Simulations for the sensitivity analysis
**Note**: to run the scripts listed here, you will need to download the necessary data from the [OSF repository](https://osf.io/mde4k/) corresponding to this project. Please download the folder ```data``` from the repository and place it in the main directory of this repository (i.e. as ```../../data```, relative to the location of this README file).

The sensitivity analysis includes three components:
1. An analysis of the change in outbreak sizes if a single parameter (effectiveness of an intervention measure) is changed.
2. An analysis of a "worst case" scenario where the effectiveness of NPIs is subject to conservative assumptions.
3. An analysis of the two vaccination screnarios (see section ```vaccinations```) under these worst case assumptions.  

## Simulations

### Individual parameters
We vary the following parameters to investigate their impact on outbreak sizes:
* The Ventilation efficiency.
* The efficiency of wearing masks.
* The ratio of agents that adhere to voluntary testing.
* The ratio of students that adhere to class size reductions.
* The sensitivity of antigen tests used for preventive screening.
* The addition of between-class contact of a ratio of students modelling friendships.
* Variations in the age dependence of the transmission risk $c_{age}$ corresponding to the 2.5 and 97.5 percentile values of this parameter from the calibration.
* Variations in the transmission risk for school contacts $c_{contact}$ corresponding to the 2.5 and 97.5 percentile values of this parameter from the calibration.
* Changed overall transmissibility of the virus (these results are not used in the publication).  

The measure combinations that are simulated for each NPI are stored in separate files for each NPI in the folder ```screening_params/```. They correspond to the screnarios shown in the publication in Figures 5 in the main manuscript and figures S4 and S5 in the supplement. If you want to simulate additional measure combinations, you can extend or replace these files with the descired measure combinations. Simulations are conducted in the script ```sensitivity_analysis_data_creation.ipynb``` with 500 iterations for each parameter combination, and stored in the folder ```../../data/sensitivity_analysis/simulation_results```.  

The simulation parameters, prevention measures and school characteristics used for these simulations are stored separately in the folder ```params```. As household transmission risk $\beta$, school transmission risk $c_{contact}$ and age dependence of the transmission risk $c_{age}$ we use the values determined during the calibration of the model (see section ```calibration```) except for simulations where these values are changed specifically.  

**Note**: The simulations for the added friendship contacts and reduced class sizes require new contact networks that reflect these changed contact patterns. These contact networks are created in the scripts  ```construct_school_networks_increased_contacts.ipynb``` and ```construct_school_networks_reduced_class_sizes.ipynb```. All other simulations require the contact networks created for the different school types in section ```intervention_measures```. We provide these contact networks in the OSF repository as compressed archives for every school type in ```../../data/contact_networks/added_friendship_contacts```, ```../../data/contact_networks/reduced_class_size``` and ```../../data/contact_networks/representative_schools```, respectively. If you want to use these pre-made contact networks, please unzip them first.

**Note**: Since running all ensembles with the necessary number of runs to create reliable statistics takes long, we provide the aggregated and extracted ensemble observables in the OSF repository at ```data/sensitivity_analysis/simulation_results/``` in a subfolder for each of the investigated parameters.

### Worst case scenario
Simulation of the worst case scenario works similar to the simulation of changes in the effectiveness of individual NPIs above. The only difference is that this time all NPIs are changed to a single (more conservatively chosen) value at once, and instead of parameter values we simulate different combinations of NPIs. Simulations are again performed in the script ```sensitivity_analysis_data_creation.ipynb``` and stored in the folder ```../../data/sensitivity_analysis/simulation_results/worst_case```.

**Note**: Again, these simulations need special contact networks because they combine the addition of contacts through friendships and removal of contacts through reduction of class sizes. These contact networks are constructed in the script ```construct_school_networks_worst_case.ipynb``` and provided as zipped archives for every school type in the OSF repository under ```../../data/contact_networks/worst_case```.

**Note**: Since running all ensembles with the necessary number of runs to create reliable statistics takes long, we provide the aggregated and extracted ensemble observables in the OSF repository at ```data/sensitivity_analysis/simulation_results/worst_case/```.

### Conservative assumptions and vaccination scenarios
Simulation of the two vaccination scenarios (see section ```vaccinations``` for details) works similar to the simulation of the worst case scenario described above. We simulate the same scenarios using the same conservative assumptions for measure effectiveness for the two vaccination scenarios. Simulations are again performed in the script ```sensitivity_analysis_data_creation.ipynb``` and stored in the folder ```../../data/sensitivity_analysis/simulation_results/worst_case_and_vaccinations```. Simulations use the contact networks created for the worst case simulations. 

**Note**: Since running all ensembles with the necessary number of runs to create reliable statistics takes long, we provide the aggregated and extracted ensemble observables in the OSF repository at ```data/sensitivity_analysis/simulation_results/worst_case_and_vaccinations/```.

## Analysis
Analysis of the ensembles of the simulation results to create figures 5 (main manuscript) and S4 and S5 (SI) is performed in the sript ```sensitivity_analysis_visualizations.ipynb```. Figures are stored at ```../../plots/sensitivity_analysis```.