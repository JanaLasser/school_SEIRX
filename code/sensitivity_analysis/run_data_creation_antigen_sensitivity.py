import networkx as nx
import pandas as pd
import numpy as np
from os.path import join
import os
import shutil
import pickle
import json

import sys
sys.path.insert(0,'../school')
sys.path.insert(0,'../nursing_home')
from model_school import SEIRX_school
import analysis_functions as af


# agents
agent_types = {
        'student':{
                'screening_interval': None, # screening param
                'index_probability': 0, # running in single index case mode
                'mask':False}, # screening param
        'teacher':{
                'screening_interval': None, # screening param
                'index_probability': 0, # running in single index case mode
                'mask':False},
        'family_member':{
                'screening_interval': None, # fixed 
                'index_probability': 0, # fixed
                'mask':False} # screening param
}

# measures
measures = {
    'testing':'preventive',
    'diagnostic_test_type':'two_day_PCR',
    'K1_contact_types':['close'],
    'quarantine_duration':10,
    'follow_up_testing_interval':None,
    'liberating_testing':False,
}

# model parameters
model_params = {
    'exposure_duration':[5.0, 1.9], # literature values
    'time_until_symptoms':[6.4, 0.8], # literature values
    'infection_duration':[10.91, 3.95], # literature values
    'subclinical_modifier':0.6, 
    'base_risk':0.0737411844049918,
    'mask_filter_efficiency':{'exhale':0.5, 'inhale':0.7},
    'infection_risk_contact_type_weights':{'very_far': 0, 'far': 0.75, 'intermediate': 0.85,'close': 1},
    'age_transmission_discount':{'slope':-0.02, 'intercept':1},
    'age_symptom_discount':{'slope':-0.02868, 'intercept':0.7954411542069012},
    'verbosity':0
}

agent_index_ratios = {
    'primary':            {'teacher':0.939394, 'student':0.060606},
    'primary_dc':         {'teacher':0.939394, 'student':0.060606},
    'lower_secondary':    {'teacher':0.568, 'student':0.432},
    'lower_secondary_dc': {'teacher':0.568, 'student':0.432},
    'upper_secondary':    {'teacher':0.182796, 'student':0.817204},
    'secondary':          {'teacher':0.362319, 'student':0.637681},
    'secondary_dc':       {'teacher':0.362319, 'student':0.637681},
}

def set_multiindex(df, agent_type):
    tuples = [(wd, a) for wd, a in zip(df['weekday'], df[agent_type])]
    index = pd.MultiIndex.from_tuples(tuples)
    df.index = index
    df = df.drop(columns=['weekday', agent_type])
    return df

def sample_prevention_strategies(screen_params, school, agent_types, measures, 
                model_params, runs, src, dst):
    # maximum number of steps in a single run. A run automatically stops if the 
    # outbreak is contained, i.e. there are no more infected or exposed agents.
    N_steps = 1000 
    
    ## data I/O
    stype = school['type']
    # construct folder for results if not yet existing
    sname = '{}_classes-{}_students-{}'.format(\
        stype, school['classes'], school['students'])

    for subdir in ['representative_runs_median', 'representative_runs_best',
                   'representative_runs_worst', 'ensembles']:
        try:
            os.mkdir(join(dst, subdir))
        except FileExistsError:
            pass   

    spath_median = join(dst, join('representative_runs_median', sname))
    spath_best = join(dst, join('representative_runs_best', sname))
    spath_worst = join(dst, join('representative_runs_worst', sname))
    spath_ensmbl = join(dst, join('ensembles', sname))

    for path in [spath_median, spath_best, spath_worst, spath_ensmbl]:
        try:
            os.mkdir(path)
        except FileExistsError:
            pass     

    node_list = pd.read_csv(join(src, '{}_node_list.csv'.format(sname)))

    ttype, index_case, s_screen_interval, t_screen_interval, student_mask, \
                teacher_mask, half_classes, ventilation_mod = screen_params
    
    turnovers = {'same':0, 'one':1, 'two':2, 'three':3}
    bmap = {True:'T', False:'F'}
    turnover, _, test = ttype.split('_')
    turnover = turnovers[turnover]
    
    measure_string = '{}_test-{}_turnover-{}_index-{}_tf-{}_sf-{}_tmask-{}'\
        .format(stype, test, turnover, index_case[0], t_screen_interval,
                s_screen_interval, bmap[teacher_mask]) +\
                '_smask-{}_half-{}_vent-{}'\
        .format(bmap[student_mask], bmap[half_classes], ventilation_mod)

    # temporary folder for all runs in the ensemble, will be
    # deleted after a representative run is picked
    tmp_path = join(spath_median, measure_string + '_tmp')
    try:
        shutil.rmtree(tmp_path)
    except FileNotFoundError:
        pass
    os.mkdir(tmp_path)
    
    half = ''
    if half_classes:
        half = '_half'
        
    # load the contact network, schedule and node_list corresponding to the school
    G = nx.readwrite.gpickle.read_gpickle(\
            join(src , '{}_network{}.bz2'.format(sname, half)))
        
    student_schedule = pd.read_csv(\
            join(src,'{}_schedule_students{}.csv'.format(sname, half)))
    student_schedule = set_multiindex(student_schedule, 'student')
    
    teacher_schedule = pd.read_csv(\
            join(src, '{}_schedule_teachers.csv'.format(sname)))
    teacher_schedule = set_multiindex(teacher_schedule, 'teacher')

    agent_types['student']['screening_interval'] = s_screen_interval
    agent_types['teacher']['screening_interval'] = t_screen_interval
    agent_types['student']['mask'] = student_mask
    agent_types['teacher']['mask'] = teacher_mask

    # results of one ensemble with the same parameters
    ensemble_results = pd.DataFrame()
    for r in range(runs):
        # instantiate model with current scenario settings
        model = SEIRX_school(G, model_params['verbosity'], 
          base_transmission_risk = model_params['base_risk'], 
          testing = measures['testing'],
          exposure_duration = model_params['exposure_duration'],
          time_until_symptoms = model_params['time_until_symptoms'],
          infection_duration = model_params['infection_duration'],
          quarantine_duration = measures['quarantine_duration'],
          subclinical_modifier = model_params['subclinical_modifier'], # literature
          infection_risk_contact_type_weights = \
                model_params['infection_risk_contact_type_weights'], # calibrated
          K1_contact_types = measures['K1_contact_types'],
          diagnostic_test_type = measures['diagnostic_test_type'],
          preventive_screening_test_type = ttype,
          follow_up_testing_interval = \
                measures['follow_up_testing_interval'],
          liberating_testing = measures['liberating_testing'],
          index_case = index_case,
          agent_types = agent_types, 
          age_transmission_risk_discount = \
                model_params['age_transmission_discount'],
          age_symptom_discount = model_params['age_symptom_discount'],
          mask_filter_efficiency = model_params['mask_filter_efficiency'],
          transmission_risk_ventilation_modifier = ventilation_mod,
          seed=r)

        # run the model, end run if the outbreak is over
        for i in range(N_steps):
            model.step()
            if len([a for a in model.schedule.agents if \
                (a.exposed == True or a.infectious == True)]) == 0:
                break

        # collect the statistics of the single run
        row = af.get_ensemble_observables_school(model, r)
        row['seed'] = r
        # add run results to the ensemble results
        ensemble_results = ensemble_results.append(row,
            ignore_index=True)
                
        # dump the current model to later pick a representative run
        N_infected = row['infected_agents']
        fname = 'run_{}_N_{}'.format(r, int(N_infected))
        af.compress_pickle(fname, tmp_path, model)


    # save the collected ensemble results
    ensemble_results.to_csv(join(spath_ensmbl, measure_string + '.csv'),
                index=False)

    # calculate ensemble statistics
    row = {'test_type':test,
           'turnover':turnover,
           'index_case':index_case,
           'student_screen_interval':s_screen_interval,
           'teacher_screen_interval':t_screen_interval,
           'student_mask':student_mask,
           'teacher_mask':teacher_mask,
           'half_classes':half_classes,
           'ventilation_modification':ventilation_mod}
    
    ensemble_results = ensemble_results[ensemble_results['infected_agents'] > 0]
    for col in ensemble_results.columns:
        row.update(af.get_statistics(ensemble_results, col))
    
    # get the a representative model with the same number of infected
    # as the ensemble median
    found = {'median':False, '0.10':False, '0.90':False}
    for stat, path in zip(['median', '0.10', '0.90'], 
                          [spath_median, spath_best, spath_worst]):

        i = 0
        while i < 10 and found[stat] == False:
            try:
                rep_model = af.get_representative_run(row['infected_agents_{}'\
                    .format(stat)], tmp_path)
                tm_events = af.get_transmission_chain(\
                            rep_model, stype, teacher_schedule, student_schedule)
                state_data = af.get_agent_states(rep_model, tm_events)

                duration = rep_model.Nstep
                start_weekday = rep_model.weekday_offset

                af.dump_JSON(path, school, ttype, index_case, s_screen_interval, 
                             t_screen_interval, teacher_mask, student_mask, 
                             half_classes, ventilation_mod, node_list, 
                             teacher_schedule, student_schedule, tm_events, 
                             state_data, start_weekday, duration)

                found[stat] = True
            except (KeyError, IndexError, AttributeError) as e:
                print('{}: {}'.format(e, measure_string))
            i += 1

    # delete the saved runs only if we found a representative run for all three
    # variants
    if all(found.values()):
        try:
            shutil.rmtree(tmp_path)
        except FileNotFoundError:
            pass

    print('completed {} {}'.format(sname, measure_string))


school_type = sys.argv[1]
runs = int(sys.argv[2])
m_idx = int(sys.argv[3]) # measure configuration index
src = sys.argv[4]
dst = sys.argv[5]

# school layouts
school_characteristics = {
    'primary':            {'classes':8, 'students':19},
    'primary_dc':         {'classes':8, 'students':19},
    'lower_secondary':    {'classes':8, 'students':18},
    'lower_secondary_dc': {'classes':8, 'students':18},
    'upper_secondary':    {'classes':10, 'students':23}, # rounded down from 10.8 classes
    'secondary':          {'classes':28, 'students':24}, # rounded up from 27.1 classes
    'secondary_dc':       {'classes':28, 'students':24} # rounded up from 27.1 classes
}
N_classes = school_characteristics[school_type]['classes']
class_size = school_characteristics[school_type]['students']
school = {'type':school_type, 'classes':N_classes,
              'students':class_size}


# screening parameters
screening_params = [ 
   # testing 1x / week for teachers AND students
  ('same_day_antigen0.1', 'student', 7, 7, False, False, False, 1),
  ('same_day_antigen0.1', 'teacher', 7, 7, False, False, False, 1),
  ('same_day_antigen0.2', 'student', 7, 7, False, False, False, 1),
  ('same_day_antigen0.2', 'teacher', 7, 7, False, False, False, 1),
  ('same_day_antigen0.3', 'student', 7, 7, False, False, False, 1),
  ('same_day_antigen0.3', 'teacher', 7, 7, False, False, False, 1),
  ('same_day_antigen0.4', 'student', 7, 7, False, False, False, 1),
  ('same_day_antigen0.4', 'teacher', 7, 7, False, False, False, 1),
  ('same_day_antigen0.5', 'student', 7, 7, False, False, False, 1),
  ('same_day_antigen0.5', 'teacher', 7, 7, False, False, False, 1),
  ('same_day_antigen0.6', 'student', 7, 7, False, False, False, 1),
  ('same_day_antigen0.6', 'teacher', 7, 7, False, False, False, 1),
  ('same_day_antigen0.7', 'student', 7, 7, False, False, False, 1),
  ('same_day_antigen0.7', 'teacher', 7, 7, False, False, False, 1),
  ('same_day_antigen0.8', 'student', 7, 7, False, False, False, 1),
  ('same_day_antigen0.8', 'teacher', 7, 7, False, False, False, 1),
  ('same_day_antigen0.9', 'student', 7, 7, False, False, False, 1),
  ('same_day_antigen0.9', 'teacher', 7, 7, False, False, False, 1),
  # ventilation + testing 1x / week for teachers AND students
  ('same_day_antigen0.1', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.1', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.2', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.2', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.3', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.3', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.4', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.4', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.5', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.5', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.6', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.6', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.7', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.7', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.8', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.8', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.9', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.9', 'teacher', 7, 7, False, False, False, 0.36),
  # ventilation + testing 2x / week for teachers AND students
  ('same_day_antigen0.1', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.1', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.2', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.2', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.3', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.3', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.4', 'student', 3, 3, False, False, False, 0.36),
  ('same_day_antigen0.4', 'teacher', 3, 3, False, False, False, 0.36),
  ('same_day_antigen0.5', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.5', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.6', 'student', 3, 3, False, False, False, 0.36),
  ('same_day_antigen0.6', 'teacher', 3, 3, False, False, False, 0.36),
  ('same_day_antigen0.7', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.7', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.8', 'student', 3, 3, False, False, False, 0.36),
  ('same_day_antigen0.8', 'teacher', 3, 3, False, False, False, 0.36),
  ('same_day_antigen0.9', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.9', 'teacher', 7, 7, False, False, False, 0.36),
  # ventilation + testing 2x / week + masks teachers AND students
  ('same_day_antigen0.1', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.1', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.2', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.2', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.3', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.3', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.4', 'student', 3, 3, True, True, False, 0.36),
  ('same_day_antigen0.4', 'teacher', 3, 3, True, True, False, 0.36),
  ('same_day_antigen0.5', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.5', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.6', 'student', 3, 3, True, True, False, 0.36),
  ('same_day_antigen0.6', 'teacher', 3, 3, True, True, False, 0.36),
  ('same_day_antigen0.7', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.7', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.8', 'student', 3, 3, True, True, False, 0.36),
  ('same_day_antigen0.8', 'teacher', 3, 3, True, True, False, 0.36),
  ('same_day_antigen0.9', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.9', 'teacher', 7, 7, False, False, False, 0.36),
  # all measures
  ('same_day_antigen0.1', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.1', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.2', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.2', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.3', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.3', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.4', 'student', 3, 3, True, True, True, 0.36),
  ('same_day_antigen0.4', 'teacher', 3, 3, True, True, True, 0.36),
  ('same_day_antigen0.5', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.5', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.6', 'student', 3, 3, True, True, True, 0.36),
  ('same_day_antigen0.6', 'teacher', 3, 3, True, True, True, 0.36),
  ('same_day_antigen0.7', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.7', 'teacher', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.8', 'student', 3, 3, True, True, True, 0.36),
  ('same_day_antigen0.8', 'teacher', 3, 3, True, True, True, 0.36),
  ('same_day_antigen0.9', 'student', 7, 7, False, False, False, 0.36),
  ('same_day_antigen0.9', 'teacher', 7, 7, False, False, False, 0.36),
]

params = screening_params[m_idx]

dst = join(dst, school_type)

try:
    os.mkdir(dst)
except FileExistsError:
    pass
    
sample_prevention_strategies(params, school, agent_types, measures, 
    model_params, runs, src, dst)




