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

## command line parameters
# school type for which the script us run.
st = sys.argv[1]
school_types = [st]
# number of simulation runs in each ensemble
N_runs = int(sys.argv[2])
    
## I/O
# source of the contact networks for the calibration runs. 
contact_network_src = '../../data/contact_networks/worst_case'
# destination of the generated data
dst = '../../data/sensitivity_analysis/simulation_results/worst_case_and_vaccinations'  

with open('params/sensitivity_analysis_measures.json', 'r') as fp:
    measures = json.load(fp)
with open('params/sensitivity_analysis_simulation_parameters.json', 'r') as fp:
    simulation_params = json.load(fp)
with open('params/sensitivity_analysis_school_characteristics.json', 'r') as fp:
    school_characteristics = json.load(fp)
    
    
# load the other screening parameters from file
screening_params = pd.read_csv(join('screening_params', 'worst_case_and_vaccinations.csv'))

params = [(N_runs, st, 
           row['index_case'],
           dcf.format_none_column(row['s_screen_interval']),
           dcf.format_none_column(row['t_screen_interval']),
           row['s_mask'],
           row['t_mask'], 
           row['class_size_reduction'],
           row['ventilation_modification'],
           row['m_efficiency_exhale'],
           row['m_efficiency_inhale'],
           row['s_test_rate'],
           row['t_test_rate'],
           row['ttype'],
           row['added_friendship_contacts'],
           row['student_vaccination_ratio'],
           row['teacher_vaccination_ratio'],
           row['family_member_vaccination_ratio']) \
           for st in school_types \
           for i, row in screening_params.iterrows()]


def run_worst_case_and_vaccinations(params):
    # extract the simulation parameters from the parameter list
    N_runs, school_type, index_case, s_screen_interval, t_screen_interval, \
        student_mask, teacher_mask, class_size_reduction, ventilation_mod, \
        mask_efficiency_exhale, mask_efficiency_inhale, s_testing_rate, \
        t_testing_rate, ttype, friendship_ratio, student_vaccination_ratio, \
        teacher_vaccination_ratio, family_member_vaccination_ratio = params
    
    try:
        os.mkdir(join(dst, school_type))
    except FileExistsError:
        pass

    # run the ensemble with the given parameter combination and school type
    ensmbl_results = dcf.run_ensemble(N_runs, school_type, measures,\
            simulation_params, school_characteristics, contact_network_src,\
            dst, index_case, s_screen_interval=s_screen_interval,
            t_screen_interval=t_screen_interval, student_mask=student_mask,
            teacher_mask=teacher_mask,class_size_reduction=class_size_reduction,
            ventilation_mod=ventilation_mod, 
            mask_efficiency_exhale=mask_efficiency_exhale,
            mask_efficiency_inhale=mask_efficiency_inhale,
            s_testing_rate=s_testing_rate, t_testing_rate=t_testing_rate,
            ttype=ttype, friendship_ratio=friendship_ratio,
            student_vaccination_ratio=student_vaccination_ratio,
            teacher_vaccination_ratio=teacher_vaccination_ratio,
            family_member_vaccination_ratio=family_member_vaccination_ratio)
    
    return ensmbl_results


# check whether we are running in test or minimal mode and reduce parameter
# list and N_runs accordingly
min_cutoff = len(params)
params = dcf.check_simulation_mode(params, N_runs, min_cutoff)
    
# run the simulation in parallel on the available cores
number_of_cores = dcf.get_number_of_cores()
pool = Pool(number_of_cores)


results = pd.DataFrame()
for ensmbl_results in tqdm(pool.imap_unordered(
    func=run_worst_case_and_vaccinations, iterable=params), total=len(params)):
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

results.to_csv(join(dst, 'worst_case_and_vaccinations_{}_{}.csv'\
                   .format(st, N_runs)), index=False)