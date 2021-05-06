# Calibration
**Note**: to run the scripts listed here, you will need to download the necessary data from the [OSF repository](https://osf.io/mde4k/) corresponding to this project. Please download the folder ```data``` from the repository and place it in the main directory of this repository (i.e. as ```../../data```, relative to the location of this README file).

Calibration of the simulation model is divided in two steps:
1. Calibration of the household transmission risk in a reduced simulation setting (only one household) and
2. Calibration of the transmission risk of intermediate and loose contacts and the dependence of transmissibility on age.

## Calibration of the household transmission risk
The calibration of the household transmission risk is performed in the script ```calibrate_household_transmission.ipynb```. The script does not require any external data to run. The only relevant simulation parameters are the distributions of exposure duration, infection duration and time until symptoms appear. The parameters defining these distributions are set in the script. The result of this script is the transmission risk per day for a contact of type "close" between two adults. Results of the simulation runs for the household risk calibration are saved in the folder ```../../data/calibration/simulation_results```.

## Calibration of contact weight and age-transmissibility dependence
### Contact network creation
Calibration simulations are run on contact networks for primary schools, lower secondary schools and secondary schools. For primary and lower secondary schools, a second version of the school with afternoon daycare is also considered ```_dc```. To this end, the contact networks of the schools first need to be created. This happens in the script ```construct_school_networks_calibration_schools.ipynb```. The contact networks created in this script are stored in the folder ```../../data/calibration/contact_network/school_type```, where ```school_type``` is the respective school type.   

Since school contact networks are to some extent random, because families are created at random and contacts between siblings can differ between two contact networks, we create 500 different contacts for every school type. Contact networks are stored as ```.bz2``` files. Every contact network also has a corresponding node list saved as ```.csv```, that lists all nodes (agents) present in the contact network. Every school type also has one schedule for teachers and students, respectively. The schedule is saved as ```.csv``` alongside the contact networks and node lists. For every agent and every hour of the (school) day, the schedule lists the room the agent is in.

### Random grid search
To calibrate the other free parameters, simulations are run for every school type on the previously created contact networks. This happens in the script ```calibration.ipynb```.  

The simulation parameters, prevention measures and school characteristics used for these simulations are stored separately in the folder ```params```. As base transmission risk, we use the value for the transmission risk of household contacts determined in the calibration of the household transmission risk above.  
The data of the empirical observations used to calibrate the simulations are stored in the folder ```../../data/calibration/empirical_observations```.

The module ```calibration_functions.py``` provides the main functions used to run the simulation and optimise the difference between simulation results and empirical observations. Simulation results from the calibration runs are stored in the folder ```../../data/calibration/simulation_results```. 

**Note**: running the at least 500 simulations necessary for the statistics to converge for each school type takes a long time on a single processor. We used a cluster to run the simulations distributed on multiple cores and aggregated the data afterwards. In the script, we set the number of runs to 1 to allow for testing of the functionality of the script, without needing a cluster. We provide the results of the random grid search and the grid search in the files ```calibration_results_random_sampling.csv``` and ```calibration_results_grid_search.csv```, respectively, in the folder ```../../data/calibration/simulation_results```. Loading these results into the script enables reproduction of our calibration results.