import numpy as np
import pandas as pd
import numpy as np
from os.path import join
import json
import calibration_functions as cf
from multiprocess import Pool
import psutil
from tqdm import tqdm


empirical_data_src = '../../data/calibration/empirical_observations'

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

N_runs = 500
#school_types = ['primary', 'primary_dc', 'lower_secondary',
#                'lower_secondary_dc', 'upper_secondary', 'secondary']
school_types = ['primary']

intermediate_contact_weights = np.arange(0, 1, 0.05)
far_contact_weights = np.arange(0, 1, 0.05)
age_transmission_discounts = np.arange(-0.1, 0, 0.02)
screening_params = [(N_runs, i, j, k, l) for i in school_types \
                    for j in intermediate_contact_weights \
                    for k in far_contact_weights \
                    for l in age_transmission_discounts if j > k]

print('There are {} parameter combinations to sample.'.format(len(screening_params)))

def run(params):
    N_runs, school_type, intermediate_contact_weight, far_contact_weight,\
                age_transmission_discount = params
    ensemble_results = cf.run_ensemble(N_runs, school_type,
            intermediate_contact_weight, far_contact_weight, 
            age_transmission_discount, prevention_measures,
            school_characteristics, agent_index_ratios,
            simulation_params, contact_network_src)
    row = cf.evaluate_ensemble(ensemble_results, school_type,
            intermediate_contact_weight, far_contact_weight,
            age_transmission_discount, outbreak_sizes, group_distributions)
    
    return row


contact_network_src = '../../data/contact_networks/calibration'

number_of_cores = psutil.cpu_count(logical=True) - 2
pool = Pool(number_of_cores)

rows = []
for row in tqdm(pool.imap_unordered(func=run, iterable=screening_params),
                total=len(screening_params)):
        rows.append(row)
pool.close()

dst = '../../data/calibration/simulation_results'
results = pd.DataFrame()
for row in rows:
    results = results.append(row, ignore_index=True)
    
results.reset_index()
index_cols = ['school_type', 'intermediate_contact_weight',
              'far_contact_weight', 'age_transmission_discount']
other_cols = [c for c in results.columns if c not in index_cols]
results = results[index_cols + other_cols]

results.to_csv(join(dst,'calibration_results_coarse_sampling_{}.csv'\
                   .format(N_runs)), index=False)
