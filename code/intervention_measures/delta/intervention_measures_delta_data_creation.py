import pandas as pd
from os.path import join
import os
import sys
import json

# custom libraries
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from data_creation_functions import run_ensemble

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
# is this a test run?
try:
    test = sys.argv[3]
    if test == 'test':
        test = True
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
dst = '../../../data/intervention_measures_delta/simulation_results'


## simulation settings
with open('params/intervention_screening_delta_measures.json', 'r') as fp:
    measures = json.load(fp)
with open('params/intervention_screening_delta_simulation_parameters.json', 'r') as fp:
    simulation_params = json.load(fp)
with open('params/intervention_screening_delta_school_characteristics.json', 'r') as fp:
    school_characteristics = json.load(fp)


## parameter grid for which simulations will be run
# specifies whether the index case will be introduced via an
# employee or a resident
index_cases = ['student', 'teacher']
# test technologies (and test result turnover times) used in the
# different scenarios
test_types = ['same_day_antigen']
# student and teacher streening intervals (in days)
s_screen_range = [None, 3, 7]
t_screen_range = [None, 3, 7]
# specifies whether teachers wear masks
student_masks = [True, False]
teacher_masks = [True, False]
half_classes = [True, False]
# specifies whether there is ventilation or not
transmission_risk_ventilation_modifiers = [1, 0.36]

params = [(N_runs, i, j, k, l, m, n, o, p, q)\
              for i in school_types \
              for j in index_cases \
              for k in test_types \
              for l in s_screen_range \
              for m in t_screen_range \
              for n in student_masks \
              for o in teacher_masks \
              for p in half_classes \
              for q in transmission_risk_ventilation_modifiers]

if test:
    params = params[0:10]
    print('This is a testrun, scanning only {} parameters with {} runs each.'\
          .format(len(params), N_runs))
else:
    print('There are {} parameter combinations to sample with {} runs each.'\
      .format(len(params), N_runs))


## simulation runs
def run(params):
    '''
    Runs an ensemble of simulations and collects observable statistics. To be 
    run in parallel on many workers. Note: I/O paths and the number of runs per 
    ensemble hare hard coded here, because I only want to pass the parameter 
    values that are being screened in the simulation run to the function via the
    parallel processing interface.
    
    Parameters:
    -----------
    param_list : iterable
        Iterable that contains the values for the parameters test_type, 
        index_case, e_screen_range and r_screen_range that are passed to the
        simulation.
        
    Returns:
    --------
    row : dictionary
        Dictionary of the ensemble statistics of the observables.
    '''    

    # extract the simulation parameters from the parameter list
    N_runs, school_type, index_case, ttype, s_screen_interval, t_screen_interval,\
        student_mask, teacher_mask, half_classes, ventilation_mod = params
    
    try:
        os.mkdir(join(dst, school_type))
    except FileExistsError:
        pass

    # run the ensemble with the given parameter combination and school type
    row = run_ensemble(N_runs, school_type, measures,\
            simulation_params, school_characteristics, contact_network_src,\
            dst, index_case, ttype, s_screen_interval, t_screen_interval,\
            student_mask, teacher_mask, half_classes, ventilation_mod)
    
    row['school_type'] = school_type
    row['index_case'] = index_case
    row['test_type'] = ttype
    row['student_screen_interval'] = s_screen_interval
    row['teacher_screen_interval'] = t_screen_interval
    row['student_mask'] = student_mask
    row['teacher_mask'] = teacher_mask
    row['half_classes'] = half_classes
    row['ventilation_mod'] = ventilation_mod
    
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
pool = Pool(number_of_cores)
rows = []
for row in tqdm(pool.imap_unordered(func=run, iterable=params), total=len(params)):
        rows.append(row)

# turn off your parallel workers 
pool.close()

# format and dump the results to disk
results = pd.DataFrame()
for row in rows:
    results = results.append(row, ignore_index=True)
    
results = results.reset_index(drop=True)
index_cols = ['school_type', 'index_case', 'test_type',
              'student_screen_interval', 'teacher_screen_interval',
              'student_mask', 'teacher_mask', 'half_classes',
              'ventilation_mod']
other_cols = [c for c in results.columns if c not in index_cols]
results = results[index_cols + other_cols]

results.to_csv(join(dst,'intervention_measures_{}_{}.csv'\
                   .format(N_runs, st)), index=False)
