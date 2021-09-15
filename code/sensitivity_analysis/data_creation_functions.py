import pandas as pd
from os.path import join
import networkx as nx
import socket
import sys

from scseirx.model_school import SEIRX_school
from scseirx import analysis_functions as af

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
                'mask':measures['student_mask'],
                'voluntary_testing_rate':1},

            'teacher':{
                'screening_interval': measures['teacher_screen_interval'],
                'index_probability': simulation_params['student_index_probability'],
                'mask':measures['teacher_mask'],
                'voluntary_testing_rate':1},

            'family_member':{
                'screening_interval':measures['family_member_screen_interval'],
                'index_probability':simulation_params['family_member_index_probability'],
                'mask':measures['family_member_mask'],
                'voluntary_testing_rate':1}
    }
    
    return agent_types


def run_model(G, agent_types, measures, simulation_params, index_case,
              base_transmission_risk_multiplier=1.0, seed=None, N_steps=1000):
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
      base_transmission_risk = \
                simulation_params['base_transmission_risk'] * \
                base_transmission_risk_multiplier,
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
      preventive_screening_test_type = \
                 measures['preventive_screening_test_type'],
      follow_up_testing_interval = \
                 measures['follow_up_testing_interval'],
      liberating_testing = measures['liberating_testing'],
      index_case = index_case,
      agent_types = agent_types, 
      age_transmission_risk_discount = \
                simulation_params['age_transmission_discount'],
      age_symptom_modification = simulation_params['age_symptom_discount'],
      mask_filter_efficiency = simulation_params['mask_filter_efficiency'],
      transmission_risk_ventilation_modifier = \
                measures['transmission_risk_ventilation_modifier'],
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
            teacher_mask=False, ventilation_mod=1.0,
            s_testing_rate=1.0, t_testing_rate=1.0, f_testing_rate=1.0,
            base_transmission_risk_multiplier=1.0,
            mask_efficiency_exhale=0.5, mask_efficiency_inhale=0.7,
            class_size_reduction=0.0, friendship_ratio=0.0,
            student_vaccination_ratio=0.0, teacher_vaccination_ratio=0.0,
            family_member_vaccination_ratio=0.0, age_transmission_discount=-0.005,
            contact_weight=0.3):
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
    agent_types['student']['voluntary_testing_rate'] = s_testing_rate
    agent_types['teacher']['voluntary_testing_rate'] = t_testing_rate
    agent_types['family_member']['voluntary_testing_rate'] = f_testing_rate
    agent_types['student']['vaccination_ratio'] = student_vaccination_ratio
    agent_types['teacher']['vaccination_ratio'] = teacher_vaccination_ratio
    agent_types['family_member']['vaccination_ratio'] = \
            family_member_vaccination_ratio
    
    simulation_params['mask_filter_efficiency']['exhale'] = \
            mask_efficiency_exhale
    simulation_params['mask_filter_efficiency']['inhale'] = \
            mask_efficiency_inhale
    simulation_params['age_transmission_discount']['slope'] = \
        age_transmission_discount
    simulation_params['infection_risk_contact_type_weights']['far'] \
        = contact_weight
    simulation_params['infection_risk_contact_type_weights']['intermediate'] \
        = contact_weight
    
    measures['preventive_screening_test_type'] = ttype
    measures['transmission_risk_ventilation_modifier'] = ventilation_mod

    sname = '{}_classes-{}_students-{}'.format(school_type,
                characteristics['classes'], characteristics['students'])
    school_src = join(contact_network_src, school_type)
    

    #print(class_size_reduction)
    #print(friendship_ratio)
    
    # load the contact network, schedule and node_list corresponding to the school
    if (class_size_reduction == 0.0) and (friendship_ratio == 0.0):
        G = nx.readwrite.gpickle.read_gpickle(\
            join(school_src, '{}_network.bz2'.format(sname)))
    elif class_size_reduction == 0.5 and (friendship_ratio == 0.0):
        # note: there are two versions of the school contact networks with half
        # of the students removed, because of how the storage of the contact 
        # networks for the sensitivity analysis is structured. We therefore
        # sequentially try to load two different contact networks, because the
        # school_src path might be different
        try:
            G = nx.readwrite.gpickle.read_gpickle(\
                join(school_src, '{}_network_half.bz2'.format(sname)))
        except FileNotFoundError:
            G = nx.readwrite.gpickle.read_gpickle(\
                join(school_src , '{}_removed-{}_network.bz2'\
               .format(sname, class_size_reduction)))
    elif class_size_reduction not in [0.0, 0.5] and (friendship_ratio == 0.0):
        G = nx.readwrite.gpickle.read_gpickle(\
            join(school_src , '{}_removed-{}_network.bz2'\
              .format(sname, class_size_reduction)))
    elif (class_size_reduction == 0) and (friendship_ratio != 0):
        try:
            G = nx.readwrite.gpickle.read_gpickle(\
                join(school_src , '{}_friends-{}_network.bz2'\
                  .format(sname, friendship_ratio)))
        except FileNotFoundError:
            G = nx.readwrite.gpickle.read_gpickle(\
                join(school_src , '{}_removed-{}_friends-{}_network.bz2'\
                  .format(sname, class_size_reduction, friendship_ratio)))
    elif (class_size_reduction == 0.5) and (friendship_ratio != 0):
        G = nx.readwrite.gpickle.read_gpickle(\
            join(school_src , '{}_friends-{}_network_half.bz2'\
              .format(sname, friendship_ratio)))
    elif (class_size_reduction == 0.3) and (friendship_ratio == 0.2):
        G = nx.readwrite.gpickle.read_gpickle(\
            join(school_src , '{}_removed-{}_friends-{}_network.bz2'\
              .format(sname, class_size_reduction, friendship_ratio)))
    else:
        print('combination of class_size_reduction and friendship_ratio ' +\
              'not supported, aborting!')
        return

    turnovers = {'same':0, 'one':1, 'two':2, 'three':3}
    bmap = {True:'T', False:'F'}
    turnover, _, test = ttype.split('_')
    turnover = turnovers[turnover]
    
    # construct the filename  and file path from the parameter values
    measure_string = '{}_test-{}_turnover-{}_index-{}_tf-{}_sf-{}_tmask-{}'\
            .format(school_type, test, turnover, index_case[0], t_screen_interval,
                    s_screen_interval, bmap[teacher_mask]) +\
                    '_smask-{}_vent-{}'\
            .format(bmap[student_mask], ventilation_mod) +\
            '_stestrate-{}_ttestrate-{}_trisk-{}_meffexh-{}_meffinh-{}'\
            .format(s_testing_rate, t_testing_rate,
                    base_transmission_risk_multiplier, mask_efficiency_exhale,
                    mask_efficiency_inhale) +\
            '_csizered-{}_fratio-{}_svacc-{}_tvacc-{}_fvacc-{}'\
            .format(class_size_reduction, friendship_ratio, 
                    student_vaccination_ratio, teacher_vaccination_ratio,
                    family_member_vaccination_ratio)
    
    if age_transmission_discount:
        measure_string += '_atd-{}'.format(age_transmission_discount)
    
    spath_ensmbl = join(res_path, school_type)
    
    # run all simulations in one ensemble (parameter combination) on one core
    ensmbl_results = pd.DataFrame()
    for r in range(1, N_runs + 1):
        
        model = run_model(
            G, agent_types, measures, simulation_params, index_case,
            base_transmission_risk_multiplier=base_transmission_risk_multiplier,
            seed=r)
        
        # collect the statistics of the single run
        row = af.get_ensemble_observables_school(model, r)
        row['seed'] = r
        # add run results to the ensemble results
        ensmbl_results = ensmbl_results.append(row,
            ignore_index=True)
    
    # add the simulation parameters to the ensemble results
    ensmbl_results['school_type'] = school_type
    ensmbl_results['index_case'] = index_case
    ensmbl_results['test_type'] = ttype
    ensmbl_results['student_screen_interval'] = s_screen_interval
    ensmbl_results['teacher_screen_interval'] = t_screen_interval
    ensmbl_results['student_mask'] = student_mask
    ensmbl_results['teacher_mask'] = teacher_mask
    ensmbl_results['ventilation_mod'] = ventilation_mod
    ensmbl_results['student_testing_rate'] = s_testing_rate
    ensmbl_results['teacher_testing_rate'] = t_testing_rate
    ensmbl_results['base_transmission_risk_multiplier'] = \
            base_transmission_risk_multiplier
    ensmbl_results['mask_efficiency_exhale'] = mask_efficiency_exhale
    ensmbl_results['mask_efficiency_inhale'] = mask_efficiency_inhale
    ensmbl_results['class_size_reduction'] = class_size_reduction
    ensmbl_results['friendship_ratio'] = friendship_ratio
    ensmbl_results['student_vaccination_ratio'] = student_vaccination_ratio
    ensmbl_results['teacher_vaccination_ratio'] = teacher_vaccination_ratio
    ensmbl_results['family_member_vaccination_ratio'] = \
            family_member_vaccination_ratio
    ensmbl_results['age_transmission_discount'] = age_transmission_discount
    ensmbl_results['contact_weight'] = contact_weight
        
    ensmbl_results.to_csv(join(spath_ensmbl, measure_string + '.csv'))
        
    return ensmbl_results


def get_number_of_cores():
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
        number_of_cores = None
        print('unknown host')
    return(number_of_cores)


def check_simulation_mode(params, N_runs, min_cutoff):
    minimum_parameters = False
    test = False
    last = False
    try:
        mod = sys.argv[3]
        # is this a test run?
        if mod == 'test':
            test = True
            N_runs = 1
            params = [(0, *p[1:]) for p in params]
        # do we just want to create the minimum necessary simulations?
        elif mod == 'min':
            minimum_parameters = True
        elif mod == 'last':
            last = True
        else:
            print('unknown command line parameter {}'.format(test))
    except IndexError:
        test = False    

    if test:
        params = params[0:10]
        print('This is a testrun, scanning only {} parameters with {} runs each.'\
              .format(len(params), N_runs))
    elif minimum_parameters:
        params = params[0:min_cutoff]
        print('Running the minimum number of necessary simulations ({})'\
              .format(len(params)))
    elif last:
        params = params[min_cutoff - 1:min_cutoff]
    else:
        print('There are {} parameter combinations to sample with {} runs each.'\
          .format(len(params), N_runs))
        
    return params
        
    
def format_none_column(x):
    if x == 'None':
        return None
    else:
        return int(x)
    

def set_measure_packages_ventilation_efficiency(data):
    # ventilation only
    data.loc[data[(data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 'never') & \
                  (data['teacher_screen_interval'] == 'never') & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation'  
    # ventilation + masks teachers + masks students + reduced class size
    data.loc[data[(data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 'never') & \
                  (data['teacher_screen_interval'] == 'never') & \
                  (data['class_size_reduction'] == 0.5)].index, 'measure'] = \
                  'ventilation + mask teachers + mask students + reduced class size'
    # ventilation + tests teachers 2x + tests students 2x
    data.loc[data[(data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation + tests teachers 2x + tests students 2x'
    # all measures
    data.loc[data[(data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.5)].index, 'measure'] = \
                  'all measures'
    
    
def set_measure_packages_mask_efficiency(data):
    # masks teachers + masks students
    data.loc[data[(data['ventilation_mod'] == 1.0) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 'never') & \
                  (data['teacher_screen_interval'] == 'never') & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'mask teachers + mask students'
    # ventilation + masks teachers + masks students + reduced class size
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 'never') & \
                  (data['teacher_screen_interval'] == 'never') & \
                  (data['class_size_reduction'] == 0.5)].index, 'measure'] = \
                  'ventilation + mask teachers + mask students + reduced class size'
    # ventilation + masks teachers AND students + tests teachers AND students 2x / week
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation + masks teachers + masks students + tests teachers 2x + tests students 2x'
    # all measures
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.5)].index, 'measure'] = \
                  'all measures'

# note: the test sensitivity simulations use the same function to assign
# measure scenarios, since the scenarios are the same
def set_measure_packages_testing_rate(data):
    # testing 1x / week for teachers AND students
    data.loc[data[(data['ventilation_mod'] == 1.0) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 7) & \
                  (data['teacher_screen_interval'] == 7) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'tests teachers 1x + tests students 1x'
    # ventilation + testing 1x / week for teachers AND students
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 7) & \
                  (data['teacher_screen_interval'] == 7) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation + tests teachers 1x + tests students 1x'
    # ventilation + testing 2x / week for teachers AND students
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation + tests teachers 2x + tests students 2x'
    # ventilation + testing 2x / week for teachers AND students + masks teachers AND students
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation + tests teachers 2x + tests students 2x + masks teachers + masks students'   
    # all measures
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.5)].index, 'measure'] = \
                  'all measures'

def set_measure_packages_reduced_class_size(data):
    # reduced_class_size
    data.loc[data[(data['ventilation_mod'] == 1.0) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 'never') & \
                  (data['teacher_screen_interval'] == 'never')].index, 'measure'] = \
                  'mask teachers + mask students'
    # ventilation + masks teachers + masks students + reduced class size
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 'never') & \
                  (data['teacher_screen_interval'] == 'never')].index, 'measure'] = \
                  'ventilation + mask teachers + mask students + reduced class size'
    # all measures
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3)].index, 'measure'] = \
                  'all measures'
    
    
# note: the transmissibility simulations use the same function to assign
# measure scenarios, since the scenarios are the same
def set_measure_packages_added_friendship_contacts(data):
    # added friendship contacts
    data.loc[data[(data['ventilation_mod'] == 1.0) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 'never') & \
                  (data['teacher_screen_interval'] == 'never') & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'no measures'
    # ventilation + masks teachers AND students + halved classes
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 'never') & \
                  (data['teacher_screen_interval'] == 'never') & \
                  (data['class_size_reduction'] == 0.5)].index, 'measure'] = \
                  'ventilation + masks teachers + masks students + halved classes'
    # ventilation + testing 2x / week teachers AND students
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation + tests teachers 2x + tests students 2x'
    # ventilation + testing 2x /week + masks teachers AND students
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation + tests teachers 2x + tests students 2x + masks teachers + masks students'   
    # all measures
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.5)].index, 'measure'] = \
                  'all measures'
    

def set_measure_packages_worst_case(data):
    # no measures
    data.loc[data[(data['ventilation_mod'] == 1) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 'never') & \
                  (data['teacher_screen_interval'] == 'never') & \
                  (data['class_size_reduction'] == 0)].index, 'measure'] = \
                  'no measures'
    # ventilation + masks teachers AND students + reduced class size
    data.loc[data[(data['ventilation_mod'] == 0.8) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 'never') & \
                  (data['teacher_screen_interval'] == 'never') & \
                  (data['class_size_reduction'] == 0.3)].index, 'measure'] = \
                  'ventilation + masks teachers + masks students + reduced class size'
    # ventilation + testing 1x / week teachers AND students
    data.loc[data[(data['ventilation_mod'] == 0.8) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 7) & \
                  (data['teacher_screen_interval'] == 7) & \
                  (data['class_size_reduction'] == 0.3)].index, 'measure'] = \
                  'ventilation + tests teachers 1x + tests students 1x'
    # ventilation + testing 2x / week teachers AND students
    data.loc[data[(data['ventilation_mod'] == 0.8) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation + tests teachers 2x + tests students 2x'
    # ventilation + testing 1x /week + masks teachers AND students
    data.loc[data[(data['ventilation_mod'] == 0.8) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 7) & \
                  (data['teacher_screen_interval'] == 7) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation + tests teachers 1x + tests students 1x + masks teachers + masks students'   
    # ventilation + testing 2x /week + masks teachers AND students
    data.loc[data[(data['ventilation_mod'] == 0.8) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation + tests teachers 2x + tests students 2x + masks teachers + masks students' 
    # ventilation + tests teachers AND students 1x / week + reduced class size
    data.loc[data[(data['ventilation_mod'] == 0.8) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 7) & \
                  (data['teacher_screen_interval'] == 7) & \
                  (data['class_size_reduction'] == 0.3)].index, 'measure'] = \
                  'ventilation + tests teachers 1x + tests students 1x + reduced class size'
    # ventilation + tests teachers AND students 2x / week + reduced class size
    data.loc[data[(data['ventilation_mod'] == 0.8) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.3)].index, 'measure'] = \
                  'ventilation + tests teachers 2x + tests students 2x + reduced class size'
    # all measures
    data.loc[data[(data['ventilation_mod'] == 0.8) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.3)].index, 'measure'] = \
                  'all measures'
    
    
def set_measure_packages_age_dependent_transmission_risk(data):
    # no measures
    data.loc[data[(data['ventilation_mod'] == 1.0) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 'never') & \
                  (data['teacher_screen_interval'] == 'never') & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'no measures'
    # ventilation + masks teachers AND students + reduced class size
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 'never') & \
                  (data['teacher_screen_interval'] == 'never') & \
                  (data['class_size_reduction'] == 0.5)].index, 'measure'] = \
                  'ventilation + masks teachers + masks students + reduced class size'
    # ventilation + testing 1x / week teachers AND students
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 7) & \
                  (data['teacher_screen_interval'] == 7) & \
                  (data['class_size_reduction'] == 0.5)].index, 'measure'] = \
                  'ventilation + tests teachers 1x + tests students 1x'
    # ventilation + testing 2x / week teachers AND students
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation + tests teachers 2x + tests students 2x'
    # ventilation + testing 1x /week + masks teachers AND students
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 7) & \
                  (data['teacher_screen_interval'] == 7) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation + tests teachers 1x + tests students 1x + masks teachers + masks students'   
    # ventilation + testing 2x /week + masks teachers AND students
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.0)].index, 'measure'] = \
                  'ventilation + tests teachers 2x + tests students 2x + masks teachers + masks students' 
    # ventilation + tests teachers AND students 1x / week + reduced class size
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 7) & \
                  (data['teacher_screen_interval'] == 7) & \
                  (data['class_size_reduction'] == 0.5)].index, 'measure'] = \
                  'ventilation + tests teachers 1x + tests students 1x + reduced class size'
    # ventilation + tests teachers AND students 2x / week + reduced class size
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == False) & \
                  (data['teacher_mask'] == False) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.5)].index, 'measure'] = \
                  'ventilation + tests teachers 2x + tests students 2x + reduced class size'
    # all measures
    data.loc[data[(data['ventilation_mod'] == 0.36) & \
                  (data['student_mask'] == True) & \
                  (data['teacher_mask'] == True) & \
                  (data['student_screen_interval'] == 3) & \
                  (data['teacher_screen_interval'] == 3) & \
                  (data['class_size_reduction'] == 0.5)].index, 'measure'] = \
                  'all measures'
    
    

