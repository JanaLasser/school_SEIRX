# Simulations for the investigation of vaccination scenarios
**Note**: to run the scripts listed here, you will need to download the necessary data from the [OSF repository](https://osf.io/mde4k/) corresponding to this project. Please download the folder ```data``` from the repository and place it in the main directory of this repository (i.e. as ```../../data```, relative to the location of this README file).

To run the simulations for the vaccination scenarios, contact networks of the schools need already be created (see the README file of the section ```intervention measures```). Once the contact networks exist, simulations are run on these contact networks to screen two vaccination scenarios: (I) with 80% vaccinated teachers, 60% vaccinated family members and 0% vaccinated students and (II) with an additional 50% vaccinated students. We simulate these two scenarios for a range of NPI combinations, assuming optimistic measure implementation stringecy .

## Running the simulations
The measure combinations that are simulated are stored in the files  ```screening_params/vaccinations.csv```. They correspond to the screnarios shown in the publication in Figure S&. If you want to simulate additional measure combinations, you can extend or replace these files with the descired measure combinations. The simulations are performed in the script ```vaccinations_data_creation.ipynb``` with 500 runs for each simulation.  

The simulation parameters, prevention measures and school characteristics used for these simulations are stored separately in the folder ```params```. As household transmission risk $\beta$, school transmission risk $c_{contact}$ and age dependence of the transmission risk $c_{age}$ we use the values determined during the calibration of the model (see section ```calibration```).  

The module ```data_creation_functions.py``` provides the main functions used to run the simulations, aggregate simulation results and extract observables. The module ```data_analysis_functions.py``` provides the functions for wrangling the data and visualizing it in plots. Simulation results are stored in the folder ```../../data/vaccinations/simulation_results```.

**Note**: Each measure combination and school type requires at least 500 runs to produce reliable statistics. This takes a long time and should therefore be run on several cores in parallel. We provide the aggregate information of the observables for every school type in the folder ```../../data/vaccintions/simulation_results```. Observables are extracted and aggregated at the end of the ```vaccinations_data_creation.ipynb``` script.

## Analysing the results
Analysis of the ensembles of simulations for each school type, parameter combination and vaccination scenario to screte figure S6 is done in the script ```vaccinations_analysis.ipynb```. Plots are stored at ```../../plots/vaccinations```.