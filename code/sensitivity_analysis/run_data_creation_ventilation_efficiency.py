import pandas as pd
import numpy as np
from os.path import join
import os
import json

from scseirx import analysis_functions as af
import data_creation_functions as dcf

# parallelisation functionality
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

# number of simulation runs in each ensemble
N_runs = int(sys.argv[2])


minimum_parameters = False
test = False
try:
    mod = sys.argv[3]
    # is this a test run?
    if mod == 'test':
        test = True
        N_runs = 1
    # do we just want to create the minimum necessary simulations?
    elif mod == 'min':
        minimum_parameters = True
    else:
        print('unknown command line parameter {}'.format(test))
except IndexError:
    test = False

    
## I/O
# source of the contact networks for the calibration runs. There is a randomly
# generated contact network for each run in the ensemble.
contact_network_src = '../../../data/contact_networks/representative_schools'
# destination of the data for the overall statistics generated in the 
# calibration run
dst = '../../data/sensitivity_analysis/simulation_results/ventilation_efficiency'  
    

with open('params/sensitivity_analysis_measures.json', 'r') as fp:
    measures = json.load(fp)
with open('params/sensitivity_analysis_simulation_parameters.json', 'r') as fp:
    simulation_params = json.load(fp)
with open('params/sensitivity_analysis_school_characteristics.json', 'r') as fp:
    school_characteristics = json.load(fp)
    

# load the other screening parameters from file and create a parameter list
screening_params = pd.read_csv(join('screening_params', 'ventilation_efficiency.csv'))
params = [(N_runs, st, 
           row['index_case'],
           format_none_column(row['s_screen_interval']),
           format_none_column(row['t_screen_interval']),
           row['s_mask'],
           row['t_mask'], 
           row['class_size_reduction'],
           row['ventilation_modification']) \
           for st in school_types \
           for i, row in screening_params.iterrows()]


if test:
    params = params[0:10]
    print('This is a testrun, scanning only {} parameters with {} runs each.'\
          .format(len(params), N_runs))
elif minimum_parameters:
    params = params[0:15]
    print('Running the minimum number of necessary simulations ({})'\
          .format(len(params)))
else:
    print('There are {} parameter combinations to sample with {} runs each.'\
      .format(len(params), N_runs))
    

## simulation runs
def run_ventilation_efficiency(params):
    '''
    Runs an ensemble of simulations and collects observable statistics. To be 
    run in parallel on many workers. Note: I/O paths and the number of runs per 
    ensemble hare hard coded here, because I only want to pass the parameter 
    values that are being screened in the simulation run to the function via the
    parallel processing interface.
    
    Parameters:
    -----------
    param_list : iterable
        Iterable that contains the values for the parameters N_runs, 
        school_type, index_case, s_screen_interval, t_screen_interval, s_mask,
        t_mask, class_size_reduction and ventilation_modification. 
        These parameters are passed to the simulation.
        
    Returns:
    --------
    row : dictionary
        Dictionary of the ensemble statistics of the observables.
    '''    

    # extract the simulation parameters from the parameter list
    N_runs, school_type, index_case, s_screen_interval, t_screen_interval, \
        student_mask, teacher_mask, class_size_reduction, ventilation_mod, \
        = params
    
    try:
        os.mkdir(join(dst, school_type))
    except FileExistsError:
        pass

    # run the ensemble with the given parameter combination and school type
    ensmbl_results = dcf.run_ensemble(N_runs, school_type, measures,\
            simulation_params, school_characteristics, contact_network_src,\
            dst, index_case, s_screen_interval=s_screen_interval,
            t_screen_interval=t_screen_interval, student_mask=student_mask,
            teacher_mask=teacher_mask, 
            class_size_reduction=class_size_reduction,
            ventilation_mod=ventilation_mod)
    
    return ensmbl_results



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
pool = Pool(number_of_cores)


results = pd.DataFrame()
for ensmbl_results in tqdm(pool.imap_unordered(func=run_ventilation_efficiency,
                        iterable=params), total=len(params)):
    results = results.append(ensmbl_results, ignore_index=True)

# turn off your parallel workers 
pool.close()
    
results = results.reset_index(drop=True)
index_cols = ['school_type', 'index_case',
              'student_screen_interval', 'teacher_screen_interval',
              'student_mask', 'teacher_mask', 'class_size_reduction',
              'ventilation_mod']
other_cols = [c for c in results.columns if c not in index_cols]
results = results[index_cols + other_cols]

results.to_csv(join(dst, 'ventilation_efficiency_{}.csv'\
                   .format(N_runs)), index=False)
