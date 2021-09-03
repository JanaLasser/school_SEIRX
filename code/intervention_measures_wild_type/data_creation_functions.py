import networkx as nx
import pandas as pd
import numpy as np
from os.path import join
import os
import shutil
import json

from scseirx.model_school import SEIRX_school
from scseirx import analysis_functions as af
from scseirx import construct_school_network as csn

def compose_agents(measures, simulation_params):
    '''
    Utility function to compose agent dictionaries as expected by the simulation
    model as input from the dictionary of prevention measures.
    
    Parameters
    ----------
    prevention_measures : dictionary
        Dictionary of prevention measures. Needs to include the fields 
        (student, teacher, family_member) _screen_interval, index_probability
        and _mask. 
        
    Returns
    -------
    agent_types : dictionary of dictionaries
        Dictionary containing the fields "screening_interval", 
        "index_probability" and "mask" for the agent groups "student", "teacher"
        and "family_member".
    
    '''
    agent_types = {
            'student':{
                'screening_interval':measures['student_screen_interval'],
                'index_probability':simulation_params['student_index_probability'],
                'mask':measures['student_mask']},

            'teacher':{
                'screening_interval': measures['teacher_screen_interval'],
                'index_probability': simulation_params['student_index_probability'],
                'mask':measures['teacher_mask']},

            'family_member':{
                'screening_interval':measures['family_member_screen_interval'],
                'index_probability':simulation_params['family_member_index_probability'],
                'mask':measures['family_member_mask']}
    }
    
    return agent_types

def run_model(G, agent_types, measures, simulation_params, index_case,
              ttype='same_day_antigen', s_screen_interval=None,
              t_screen_interval=None, student_mask=False, 
              teacher_mask=False, half_classes=False, ventilation_mod=1,
              seed=None, N_steps=1000):
    '''
    Runs a simulation with an SEIRX_school model 
    (see https://pypi.org/project/scseirx/1.3.0/), given a set of parameters 
    which are calibrated.
    
    Parameters:
    -----------
    G : networkx Graph
        Contact network of the given school.
    agent_types : dict
        Dictionary of dictionaries, holding agent-specific information for each
        agent group.
    measures : dictionary
        Dictionary listing all prevention measures in place for the given
        scenario. Fields that are not specifically included in this dictionary
        will revert to SEIRX_school defaults.
    simulation_params : dictionary
        Dictionary holding simulation parameters such as "verbosity" and
        "base_transmission_risk". Fields that are not included will revert back
        to SEIRX_school defaults.
    index_case : string
        Agent group from which the index case is drawn. Can be "student" or
        "teacher".
    ttype : string
        Test type used for preventive screening. For example "same_day_antigen"
    s_screen_interval : integer
        Interval between preventive screens in the student agent group.
    t_screen_interval : integer
        Interval between preventive screens in the teacher agent group.
    student_mask : bool
        Wheter or not students wear masks.
    teacher_mask : bool
        Wheter or not teachers wear masks.
    half_classes : bool
        Wheter or not class sizes are reduced.
    ventilation_mod : float
        Modification to the transmission risk due to ventilation. 
        1 = no modification.
    seed : integer
        Seed for the simulation to fix randomness.
    N_steps : integer
        Number of maximum steps per run. This is a very conservatively chosen 
        value that ensures that an outbreak will always terminate within the 
        allotted time. Most runs are terminated way earlier anyways, as soon as 
        the outbreak is over.
        
    Returns
    -------
    model : SEIRX_school model instance holding a completed simulation run and
        all associated data.
    '''

    # initialize the model
    model = SEIRX_school(G, 
      simulation_params['verbosity'], 
      base_transmission_risk = simulation_params['base_transmission_risk'],
      testing = measures['testing'],
      exposure_duration = simulation_params['exposure_duration'],
      time_until_symptoms = simulation_params['time_until_symptoms'],
      infection_duration = simulation_params['infection_duration'],
      quarantine_duration = measures['quarantine_duration'],
      subclinical_modifier = simulation_params['subclinical_modifier'],
      infection_risk_contact_type_weights = \
                 simulation_params['infection_risk_contact_type_weights'],
      K1_contact_types = measures['K1_contact_types'],
      diagnostic_test_type = measures['diagnostic_test_type'],
      preventive_screening_test_type = ttype,
      follow_up_testing_interval = \
                 measures['follow_up_testing_interval'],
      liberating_testing = measures['liberating_testing'],
      index_case = index_case,
      agent_types = agent_types, 
      age_transmission_risk_discount = \
                         simulation_params['age_transmission_discount'],
      age_symptom_modification = simulation_params['age_symptom_discount'],
      mask_filter_efficiency = simulation_params['mask_filter_efficiency'],
      transmission_risk_ventilation_modifier = ventilation_mod,
      seed=seed)

    # run the model until the outbreak is over
    for i in range(N_steps):
        # break if first outbreak is over
        if len([a for a in model.schedule.agents if \
            (a.exposed == True or a.infectious == True)]) == 0:
            break
        model.step()
        
    return model

def run_ensemble(N_runs, school_type, measures, simulation_params,
              school_characteristics, contact_network_src, res_path, index_case,
              ttype='same_day_antigen', s_screen_interval=None,
              t_screen_interval=None, student_mask=False, 
              teacher_mask=False, half_classes=False, ventilation_mod=1,):
    '''
    Utility function to run an ensemble of simulations for a given school type
    and parameter combination.
    
    Parameters:
    ----------
    N_runs : integer
        Number of individual simulation runs in the ensemble.
        school_type : string
        School type for which the model is run. This affects the selected school
        characteristics and ratio of index cases between students and teachers.
        Can be "primary", "primary_dc", "lower_secondary", "lower_secondary_dc",
        "upper_secondary", "secondary" or "secondary_dc".
    school_type : string
        School type for which the ensemble is run. This affects the selected 
        school characteristics and ratio of index cases between students and 
        teachers. Can be "primary", "primary_dc", "lower_secondary", 
        "lower_secondary_dc", "upper_secondary", "secondary" or "secondary_dc".
    measures : dictionary
        Dictionary listing all prevention measures in place for the given
        scenario. Fields that are not specifically included in this dictionary
        will revert to SEIRX_school defaults.
    simulation_params : dictionary
        Dictionary holding simulation parameters such as "verbosity" and
        "base_transmission_risk". Fields that are not included will revert back
        to SEIRX_school defaults.
    school_characteristics : dictionary
        Dictionary holding the characteristics of each possible school type. 
        Needs to include the fields "classes" and "students" (i.e. the number)
        of students per class. The number of teachers is calculated
        automatically from the given school type and number of classes.
    res_path : string
        Path to the directory in which results will be saved.
    contact_network_src : string
        Absolute or relative path pointing to the location of the contact
        network used for the calibration runs. The location needs to hold the
        contact networks for each school types in a sub-folder with the same
        name as the school type. Networks need to be saved in networkx's .bz2
        format.
    index_case : string
        Agent group from which the index case is drawn. Can be "student" or
        "teacher".
    ttype : string
        Test type used for preventive screening. For example "same_day_antigen"
    s_screen_interval : integer
        Interval between preventive screens in the student agent group.
    t_screen_interval : integer
        Interval between preventive screens in the teacher agent group.
    student_mask : bool
        Wheter or not students wear masks.
    teacher_mask : bool
        Wheter or not teachers wear masks.
    half_classes : bool
        Wheter or not class sizes are reduced.
    ventilation_mod : float
        Modification to the transmission risk due to ventilation. 
        1 = no modification.
        
    Returns:
    --------
    ensemble_results : pandas DataFrame
        Data Frame holding the observable of interest of the ensemble, namely
        the number of infected students and teachers.
    '''
    characteristics = school_characteristics[school_type]
    # create the agent dictionaries based on the given parameter values and
    # prevention measures
    agent_types = compose_agents(measures, simulation_params)
    agent_types['student']['screening_interval'] = s_screen_interval
    agent_types['teacher']['screening_interval'] = t_screen_interval
    agent_types['student']['mask'] = student_mask
    agent_types['teacher']['mask'] = teacher_mask

    sname = '{}_classes-{}_students-{}'.format(school_type,
                characteristics['classes'], characteristics['students'])
    school_src = join(contact_network_src, school_type)
    
    half = ''
    if half_classes:
        half = '_half'

    # load the contact network, schedule and node_list corresponding to the school
    G = nx.readwrite.gpickle.read_gpickle(\
            join(school_src, '{}_network{}.bz2'.format(sname, half)))    

    turnovers = {'same':0, 'one':1, 'two':2, 'three':3}
    bmap = {True:'T', False:'F'}
    turnover, _, test = ttype.split('_')
    turnover = turnovers[turnover]
        
    measure_string = '{}_test-{}_turnover-{}_index-{}_tf-{}_sf-{}_tmask-{}'\
            .format(school_type, test, turnover, index_case[0], t_screen_interval,
                    s_screen_interval, bmap[teacher_mask]) +\
                    '_smask-{}_half-{}_vent-{}'\
            .format(bmap[student_mask], bmap[half_classes], ventilation_mod)
    
    spath_ensmbl = join(res_path, school_type)
    
    ensemble_results = pd.DataFrame()
    for r in range(1, N_runs + 1):
        model = run_model(G, agent_types, measures,simulation_params,index_case,
              ttype, s_screen_interval, t_screen_interval, student_mask, 
              teacher_mask, half_classes, ventilation_mod, seed=r)
        
        # collect the statistics of the single run
        row = af.get_ensemble_observables_school(model, r)
        row['seed'] = r
        # add run results to the ensemble results
        ensemble_results = ensemble_results.append(row,
            ignore_index=True)
        
    ensemble_results.to_csv(join(spath_ensmbl, measure_string + '.csv'))
        
    return ensemble_results   

def get_data(stype, src_path):
    data = pd.DataFrame()
    stype_path = join(src_path, stype)
    files = os.listdir(stype_path)
    for f in files:
        screening_params, agents, half = af.get_measures(f.strip('.csv'))
        ensmbl = pd.read_csv(join(stype_path, f))
        try:
            ensmbl = ensmbl.drop(columns=['Unnamed: 0'])
        except KeyError:
            pass
        ensmbl['preventive_test_type'] = screening_params['preventive_test_type']
        ensmbl['index_case'] = screening_params['index_case']
        ensmbl['transmission_risk_ventilation_modifier'] = \
            screening_params['transmission_risk_ventilation_modifier']
        ensmbl['student_mask'] = agents['student']['mask']
        ensmbl['teacher_mask'] = agents['teacher']['mask']
        ensmbl['student_screening_interval'] = agents['student']['screening_interval']
        ensmbl['teacher_screening_interval'] = agents['teacher']['screening_interval']
        ensmbl['half_classes'] = half

        data = pd.concat([data, ensmbl])

    data = data.reset_index(drop=True)
    data['teacher_screening_interval'] = data['teacher_screening_interval']\
        .replace({None:'never'})
    data['student_screening_interval'] = data['student_screening_interval']\
        .replace({None:'never'})
    return data

def set_individual_measures(data):
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 1) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screening_interval'] == 'never') & \
                  (data['teacher_screening_interval'] == 'never') & \
                  (data['half_classes'] == False)].index, 'measure'] = 'no\nmeasure'
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 1) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screening_interval'] == 'never') & \
                  (data['teacher_screening_interval'] == 'never') & \
                  (data['half_classes'] == False)].index, 'measure'] = 'mask\nteacher'
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 1) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screening_interval'] == 'never') & \
                  (data['teacher_screening_interval'] == 'never') & \
                  (data['half_classes'] == False)].index, 'measure'] = 'mask\nstudent'
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screening_interval'] == 'never') & \
                  (data['teacher_screening_interval'] == 'never') & \
                  (data['half_classes'] == False)].index, 'measure'] = 'ventilation'
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 1) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screening_interval'] == 'never') & \
                  (data['teacher_screening_interval'] == 'never') & \
                  (data['half_classes'] == True)].index, 'measure'] = 'halved\nclasses'
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 1) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screening_interval'] == 7) & \
                  (data['teacher_screening_interval'] == 'never') & \
                  (data['half_classes'] == False)].index, 'measure'] = 'student tests\n1x / week'
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 1) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screening_interval'] == 3) & \
                  (data['teacher_screening_interval'] == 'never') & \
                  (data['half_classes'] == False)].index, 'measure'] = 'student tests\n2x / week'
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 1) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screening_interval'] == 'never') & \
                  (data['teacher_screening_interval'] == 7) & \
                  (data['half_classes'] == False)].index, 'measure'] = 'teacher tests\n1x / week'
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 1) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screening_interval'] == 'never') & \
                  (data['teacher_screening_interval'] == 3) & \
                  (data['half_classes'] == False)].index, 'measure'] = 'teacher tests\n2x / week'
    
    
def set_measure_packages(data):
    # ventilation + masks teachers
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screening_interval'] == 'never') & \
                  (data['teacher_screening_interval'] == 'never') & \
                  (data['half_classes'] == False)].index, 'measure'] = \
                  'ventilation + mask teachers'
    # ventilation + masks teachers + masks students
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screening_interval'] == 'never') & \
                  (data['teacher_screening_interval'] == 'never') & \
                  (data['half_classes'] == False)].index, 'measure'] = \
                  'ventilation + mask teachers + mask students'
    # ventilation + masks teachers + masks students + halved classes
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screening_interval'] == 'never') & \
                  (data['teacher_screening_interval'] == 'never') & \
                  (data['half_classes'] == True)].index, 'measure'] = \
                  'ventilation + mask teachers + mask students + halved classes'
    
    # ventilation + tests teachers 1x
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screening_interval'] == 'never') & \
                  (data['teacher_screening_interval'] == 7) & \
                  (data['half_classes'] == False)].index, 'measure'] = \
                  'ventilation + tests teachers 1x'
    # ventilation + tests teachers 1x + tests students 1x
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screening_interval'] == 7) & \
                  (data['teacher_screening_interval'] == 7) & \
                  (data['half_classes'] == False)].index, 'measure'] =\
                  'ventilation + tests teachers 1x + tests students 1x'
    # ventilation + tests teachers 2x + tests students 1x
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screening_interval'] == 7) & \
                  (data['teacher_screening_interval'] == 3) & \
                  (data['half_classes'] == False)].index, 'measure'] = \
                  'ventilation + tests teachers 2x + tests students 1x'
    # ventilation + tests teachers 2x + tests students 2x
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screening_interval'] == 3) & \
                  (data['teacher_screening_interval'] == 3) & \
                  (data['half_classes'] == False)].index, 'measure'] = \
                  'ventilation + tests teachers 2x + tests students 2x'
    
    # ventilation + masks teachers & students + tests 1x teachers & students
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screening_interval'] == 7) & \
                  (data['teacher_screening_interval'] == 7) & \
                  (data['half_classes'] == False)].index, 'measure'] = \
                  'ventilation + masks teachers & students + tests 1x teachers & students'
    # ventilation + halved classes + tests 1x teachers & students
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screening_interval'] == 7) & \
                  (data['teacher_screening_interval'] == 7) & \
                  (data['half_classes'] == True)].index, 'measure'] = \
                  'ventilation + halved classes + tests 1x teachers & students'
    
    # all measures
    data.loc[data[(data['transmission_risk_ventilation_modifier'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screening_interval'] == 3) & \
                  (data['teacher_screening_interval'] == 3) & \
                  (data['half_classes'] == True)].index, 'measure'] = \
                  'all measures'