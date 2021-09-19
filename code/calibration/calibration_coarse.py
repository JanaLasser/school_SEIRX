import numpy as np
import pandas as pd
import numpy as np
from os.path import join
import json
import calibration_functions as cf
from multiprocess import Pool
import psutil
from tqdm import tqdm
import sys
import socket

## command line parameters
# school type for which the script us run. The final optimization combines
# results from all six school types. We split ensemble runs into the different
# school types to use all available computational resources.
st = sys.argv[1]
school_types = [st]

# optimal value for the intermediate and far contact weights, determined in the
# coarse optimization run with small (N=500) ensembles that preceded this run.
opt_contact_weight_prev = sys.argv[2]

# number of simulation runs in each ensemble
N_runs = int(sys.argv[3])
# is this a test run?
try:
    test = sys.argv[4]
    if test == 'test':
        test = True
    else:
        print('unknown command line parameter {}'.format(test))
except IndexError:
    test = False

## I/O
# source of the contact networks for the calibration runs. There is a randomly
# generated contact network for each run in the ensemble.
contact_network_src = '../../data/contact_networks/calibration'
# destination for the data of every single run in the ensemble that will 
# generated and stored during the simulation 
ensmbl_dst = '../../data/calibration/simulation_results/ensembles_coarse'
# destination of the data for the overall statistics generated in the 
# calibration run
dst = '../../data/calibration/simulation_results'
# source of the empirically observed outbreak data
empirical_data_src = '../../data/calibration/empirical_observations'


## empirical outbreak data
# load the empirically observed outbreak size distributions with which we 
# compare the outbreak size distributions of the simulated ensembles. 
outbreak_sizes = pd.read_csv(\
            join(empirical_data_src, 'empirical_outbreak_sizes.csv'))
group_distributions = pd.read_csv(\
            join(empirical_data_src, 'empirical_group_distributions.csv'))
agent_index_ratios = pd.read_csv(\
            join(empirical_data_src, 'empirical_index_case_ratios.csv'))
agent_index_ratios.index = agent_index_ratios['school_type']
symptomatic_case_ratios = pd.read_csv(\
            join(empirical_data_src, 'empirical_symptomatic_case_ratios.csv'))

with open('params/calibration_measures.json', 'r') as fp:
    prevention_measures = json.load(fp)
with open('params/calibration_simulation_parameters.json', 'r') as fp:
    simulation_params = json.load(fp)
with open('params/calibration_school_characteristics.json', 'r') as fp:
    school_characteristics = json.load(fp)


## parameter grid for which simulations will be run

opt_contact_weight_coarse = float(opt_contact_weight_coarse)
# the contact weight is the modifier by which the base transmission risk (for
# household transmissions) is multiplied for contacts of type "intermediate" 
# and of type "far". Parameter values are chosen around the optimum from the
# previous random sampling search, passed to the script via the command line.
contact_weights_coarse = np.hstack([
    np.arange(opt_contact_weight_prev - 0.05, 
              opt_contact_weight_prev, 0.01),
    np.arange(opt_contact_weight_prev, 
              opt_contact_weight_prev + 0.06, 0.01)
    ])

# the age_transmission_discount sets the slope of the age-dependence of the 
# transmission risk. Transmission risk for adults (age 18+) is always base 
# transmission risk. For every year an agent is younger than 18 years, the
# transmission risk is reduced. Parameter values are chosen around the optimum 
# from the previous random sampling search
age_transmission_discounts_coarse = [-0.01,-0.02,-0.03,-0.04,-0.05,-0.06, 
                                   -0.07, -0.08, -0.09]

contact_weights_coarse = np.asarray([round(i, 2) \
            for i in contact_weights_coarse])


print('contact weiths: ', contact_weights_coarse)
print('age discounts: ', age_transmission_discounts_coarse)

# list of all possible parameter combinations from the grid
# Note: the age transmission discount is set to 0 for all parameter
# combinations here
screening_params = [(N_runs, i, j, j, k) for i in school_types \
                    for j in contact_weights_coarse \
                    for k in age_transmission_discounts_coarse]

if test:
    screening_params = screening_params[0:10]
    print('This is a testrun, scanning only {} parameters with {} runs each.'\
          .format(len(screening_params), N_runs))
else:
    print('There are {} parameter combinations to sample with {} runs each.'\
      .format(len(screening_params), N_runs))


## simulation runs
def run(params):
    N_runs, school_type, intermediate_contact_weight, far_contact_weight,\
                age_transmission_discount = params
    ensemble_results = cf.run_ensemble(N_runs, school_type,
            intermediate_contact_weight, far_contact_weight, 
            age_transmission_discount, prevention_measures,
            school_characteristics, agent_index_ratios,
            simulation_params, contact_network_src, ensmbl_dst)
    row = cf.evaluate_ensemble(ensemble_results, school_type,
            intermediate_contact_weight, far_contact_weight,
            age_transmission_discount, outbreak_sizes, group_distributions)
    
    return row

# figure out which host we are running on and determine number of cores to
# use for the parallel programming
hostname = socket.gethostname()
if hostname == 'desiato':
    number_of_cores = 200 # desiato
    print('running on {}, using {} cores'.format(hostname, number_of_cores))
elif hostname == 'T14s':
    number_of_cores = 14 # laptop
    print('running on {}, using {} cores'.format(hostname, number_of_cores))
elif hostname == 'marvin':
    number_of_cores = 28 # marvin
    print('running on {}, using {} cores'.format(hostname, number_of_cores))
else:
    print('unknown host')

# run the simulation in parallel on the available cores
rows = []
pool = Pool(number_of_cores)
for row in tqdm(pool.imap_unordered(func=run, iterable=screening_params),
                total=len(screening_params)):
        rows.append(row)
pool.close()

# collect and save the results
results = pd.DataFrame()
for row in rows:
    results = results.append(row, ignore_index=True)
    
results.reset_index()
index_cols = ['school_type', 'intermediate_contact_weight',
              'far_contact_weight', 'age_transmission_discount']
other_cols = [c for c in results.columns if c not in index_cols]
results = results[index_cols + other_cols]

results.to_csv(join(dst,'calibration_results_coarse_sampling_{}_{}.csv'\
                   .format(N_runs, st)), index=False)
