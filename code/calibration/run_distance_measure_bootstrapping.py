import calibration_functions as cf
import pandas as pd
from os.path import join
import socket
import sys
import numpy as np
from multiprocess import Pool
import psutil
from tqdm import tqdm

# empirical observations
empirical_data_src = '../../data/calibration/empirical_observations'
outbreak_sizes = pd.read_csv(\
            join(empirical_data_src, 'empirical_outbreak_sizes.csv'))
group_distributions = pd.read_csv(\
            join(empirical_data_src, 'empirical_group_distributions.csv'))

# screening params
opt_contact_weight_coarse = 0.29
N_runs = 4000

school_types = ['primary', 'primary_dc', 'lower_secondary',
                'lower_secondary_dc', 'upper_secondary', 'secondary']
contact_weights_fine = np.hstack([
    np.arange(opt_contact_weight_coarse - 0.06, 
              opt_contact_weight_coarse, 0.01),
    np.arange(opt_contact_weight_coarse, 
              opt_contact_weight_coarse + 0.061, 0.01)
    ])
age_transmission_discounts_fine = [0.00, -0.0025, -0.005, -0.0075, -0.01,
                                   -0.0125, -0.015, -0.0175, -0.02,
                                   -0.0225, -0.025, -0.0275, -0.03]

contact_weights_fine = np.asarray([round(i, 2) \
            for i in contact_weights_fine])

screening_params = [(N_runs, i, j, j, k) for i in school_types \
                    for j in contact_weights_fine \
                    for k in age_transmission_discounts_fine]


print('There are {} parameter combinations to sample with {} runs each.'\
  .format(len(screening_params), N_runs))


def run_all_distances(params):
    ensemble_results, st, icw, fcw, atd, outbreak_sizes, \
    group_distributions, bootstrap_run = params
    
    row = cf.calculate_distances(ensemble_results, st, icw, fcw, atd,
                       outbreak_sizes, group_distributions)
    
    row.update({'bootstrap_run':bootstrap_run})
    
    return row

# calculate the various distribution distances between the simulated and
# observed outbreak size distributions
src = '../../data/calibration/simulation_results/ensembles_fine_ensemble_distributions'
dst = '../../data/calibration/simulation_results/'
N_bootstrap = int(sys.argv[1])

bootstrapping_results = pd.DataFrame()
for i, ep in enumerate(screening_params):
    _, school_type, icw, fcw, atd = ep
    if i % 100 == 0:
        print('{}/{}'.format(i, len(screening_params)))
        
    fname = 'school_type-{}_icw-{:1.2f}_fcw-{:1.2f}_atd-{:1.4f}_infected.csv'\
        .format(school_type, icw, fcw, atd)
    ensemble_results = pd.read_csv(join(src, fname), 
            dtype={'infected_students':int, 'infected_teachers':int,
                   'infected_total':int, 'run':int})
    
    bootstrap_params = [(ensemble_results.sample(2000), school_type, icw, fcw, \
                        atd, outbreak_sizes, group_distributions, j) \
                        for j in range(N_bootstrap)]
    
    hostname = socket.gethostname()
    if hostname == 'desiato':
        number_of_cores = 250 # desiato
        print('running on {}, using {} cores'.format(hostname, number_of_cores))
    elif hostname == 'T14s':
        number_of_cores = 14 # laptop
        print('running on {}, using {} cores'.format(hostname, number_of_cores))
    elif hostname == 'marvin':
        number_of_cores = 28 # marvin
        print('running on {}, using {} cores'.format(hostname, number_of_cores))
    else:
        print('unknown host')
    
    pool = Pool(number_of_cores)

    for res in tqdm(pool.imap_unordered(func=run_all_distances,
                    iterable=bootstrap_params), total=len(bootstrap_params)):
        bootstrapping_results = bootstrapping_results.append(res, ignore_index=True)
        
    pool.close()
    
bootstrapping_results.to_csv(join(dst, 'bootstrapping_results_{}.csv'\
                                .format(N_bootstrap)), index=False)