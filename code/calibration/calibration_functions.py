from scseirx.model_school import SEIRX_school
import scseirx.analysis_functions as af
import pandas as pd
import numpy as np
import networkx as nx
from os.path import join

def calculate_distribution_difference(school_type, ensemble_results, \
                                      outbreak_sizes):
    '''
    Calculates the difference between the expected distribution of outbreak
    sizes and the observed outbreak sizes in an ensemble of simulation runs
    with the same parameters. The data-frame ensemble_results holds the number
    of infected students and the number of infected teachers. NOTE: the index
    case is already subtracted from these numbers.
    
    Parameters
    ----------
    school_type : string
        school type for which the distribution difference should be calculated.
        Can be "primary", "primary_dc", "lower_secondary", "lower_secondary_dc",
        "upper_secondary", "secondary" or "secondary_dc"
    ensemble_results : pandas DataFrame
        Data frame holding the results of the simulated outbreaks for a given
        school type and parameter combination. The outbreak size has to be given
        in the column "infected_total". 
    outbreak_size : pandas DataFrame
        Data frame holding the empirical outbreak size observations. The 
        outbreak size has to be given in the column "size", the school type in
        the column "type".
        
    Returns
    -------
    chi_2_distance : float
        chi-squared distance between the simulated and empirically observed
        outbreak size distributions
    sum_of_squares : float
        sum of squared differences between the simulated and empirically 
        observed outbreak size distributions.
    '''
    
    # calculate the total number of follow-up cases (outbreak size)
    ensemble_results['infected_total'] = ensemble_results['infected_teachers'] +\
                    ensemble_results['infected_students']
    
    ensemble_results = ensemble_results.astype(int)
    
    # censor runs with no follow-up cases as we also do not observe these in the
    # empirical data
    ensemble_results = ensemble_results[ensemble_results['infected_total'] > 0].copy()
    observed_outbreaks = ensemble_results['infected_total'].value_counts()
    observed_outbreaks = observed_outbreaks / observed_outbreaks.sum()
    obs_dict = {size:ratio for size, ratio in zip(observed_outbreaks.index,
                                                   observed_outbreaks.values)}
    
    # since we only have aggregated data for schools with and without daycare,
    # we map the daycare school types to their corresponding non-daycare types,
    # which are also the labels of the schools in the emirical data
    type_map = {'primary':'primary', 'primary_dc':'primary',
                'lower_secondary':'lower_secondary',
                'lower_secondary_dc':'lower_secondary',
                'upper_secondary':'upper_secondary',
                'secondary':'secondary', 'secondary_dc':'secondary'}
    school_type = type_map[school_type]

    expected_outbreaks = outbreak_sizes[\
                            outbreak_sizes['type'] == school_type].copy()
    expected_outbreaks.index = expected_outbreaks['size']
    
    exp_dict = {s:c for s, c in zip(expected_outbreaks.index, 
                                     expected_outbreaks['ratio'])}
    
    # add zeroes for both the expected and observed distributions in cases 
    # (sizes) that were not observed
    if len(observed_outbreaks) == 0:
        obs_max = 0
    else:
        obs_max = observed_outbreaks.index.max()
    
    for i in range(1, max(obs_max + 1,
                          expected_outbreaks.index.max() + 1)):
        if i not in observed_outbreaks.index:
            obs_dict[i] = 0
        if i not in expected_outbreaks.index:
            exp_dict[i] = 0
            
    obs = np.asarray([obs_dict[i] for i in range(1, len(obs_dict) + 1)])
    exp = np.asarray([exp_dict[i] for i in range(1, len(exp_dict) + 1)])
    
    chi2_distance = ((exp + 1) - (obs + 1))**2 / (exp + 1)
    chi2_distance = chi2_distance.sum()
    
    sum_of_squares = ((exp - obs)**2).sum()
    
    return chi2_distance, sum_of_squares


def calculate_group_case_difference(school_type, ensemble_results,\
                                   group_distributions):
    '''
    Calculates the difference between the expected number of infected teachers
    / infected students and the observed number of infected teachers / students
    in an ensemble of simulation runs with the same parameters. The data-frame 
    ensemble_results holds the number of infected students and the number of 
    infected teachers. NOTE: the index case is already subtracted from these
    numbers.
    
    Parameters
    ----------
    school_type : string
        school type for which the distribution difference should be calculated.
        Can be "primary", "primary_dc", "lower_secondary", "lower_secondary_dc",
        "upper_secondary", "secondary" or "secondary_dc"
    ensemble_results : pandas DataFrame
        Data frame holding the results of the simulated outbreaks for a given
        school type and parameter combination. The outbreak size has to be given
        in the column "infected_total". 
    group_distributions : pandas DataFrame
        Data frame holding the empirical observations of the ratio of infections
        in a given group (student, teacher) as compared to the overall number of
        infections (students + teachers). The data frame has three columns:
        "school_type", "group" and "ratio", where "group" indicates which group
        (student or teacher) the number in "ratio" belongs to. 
        
    Returns
    -------
    chi_2_distance : float
        chi-squared distance between the simulated and empirically observed
        outbreak size distributions
    sum_of_squares : float
        sum of squared differences between the simulated and empirically 
        observed outbreak size distributions.
    '''
    
    # calculate the total number of follow-up cases (outbreak size)
    ensemble_results['infected_total'] = ensemble_results['infected_teachers'] +\
                    ensemble_results['infected_students']
    
    # censor runs with no follow-up cases as we also do not observe these in the
    # empirical data
    ensemble_results = ensemble_results[ensemble_results['infected_total'] > 0].copy()
    
    # calculate ratios of infected teachers and students
    ensemble_results['teacher_ratio'] = ensemble_results['infected_teachers'] / \
                                        ensemble_results['infected_total'] 
    ensemble_results['student_ratio'] = ensemble_results['infected_students'] / \
                                        ensemble_results['infected_total'] 
    
    observed_distro = pd.DataFrame({'group':['student', 'teacher'],
                                    'ratio':[ensemble_results['student_ratio'].mean(),
                                             ensemble_results['teacher_ratio'].mean()]})

    # since we only have aggregated data for schools with and without daycare,
    # we map the daycare school types to their corresponding non-daycare types,
    # which are also the labels of the schools in the emirical data
    type_map = {'primary':'primary', 'primary_dc':'primary',
                'lower_secondary':'lower_secondary',
                'lower_secondary_dc':'lower_secondary',
                'upper_secondary':'upper_secondary',
                'secondary':'secondary', 'secondary_dc':'secondary'}
    school_type = type_map[school_type]
    
    expected_distro = group_distributions[\
                                group_distributions['type'] == school_type].copy()
    expected_distro.index = expected_distro['group']
    
    obs = observed_distro['ratio'].values
    exp = expected_distro['ratio'].values
    
    chi2_distance = ((exp + 1) - (obs + 1))**2 / (exp + 1)
    chi2_distance = chi2_distance.sum()
    
    sum_of_squares = ((exp - obs)**2).sum()
    
    return chi2_distance, sum_of_squares


def compose_agents(prevention_measures):
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
                'screening_interval':prevention_measures['student_screen_interval'],
                'index_probability':prevention_measures['student_index_probability'],
                'mask':prevention_measures['student_mask']},

            'teacher':{
                'screening_interval': prevention_measures['teacher_screen_interval'],
                'index_probability': prevention_measures['student_index_probability'],
                'mask':prevention_measures['teacher_mask']},

            'family_member':{
                'screening_interval':prevention_measures['family_member_screen_interval'],
                'index_probability':prevention_measures['family_member_index_probability'],
                'mask':prevention_measures['family_member_mask']}
    }
    
    return agent_types


def run_model(school_type, run, intermediate_contact_weight,
              far_contact_weight, age_transmission_discount, 
              prevention_measures, school_characteristics,
              agent_index_ratios, simulation_params,
              contact_network_src, N_steps=500):
    '''
    Runs a simulation with an SEIRX_school model 
    (see https://pypi.org/project/scseirx/1.3.0/), given a set of parameters 
    which are calibrated.
    
    Parameters:
    -----------
    school_type : string
        School type for which the model is run. This affects the selected school
        characteristics and ratio of index cases between students and teachers.
        Can be "primary", "primary_dc", "lower_secondary", "lower_secondary_dc",
        "upper_secondary", "secondary" or "secondary_dc".
    run : integer
        Consecutive number of the simulation run within the ensemble of 
        simulation runs with the same school type and parameter combination.
        This is needed to load the correct contact network, since every run
        in a given ensemble uses a different contact network that has a random
        choice of household sizes and sibling connections, based on the 
        Austrian household statistics.
    intermediate_contact_weight : float
        Weight of contacts of type "intermediate" (as compared to household)
        contacts. Note: This parameter is formulated in terms of a "weight",
        i.e. a multiplicative factor to the intensity of the household contact
        (which is 1 by default). This is different from the "probability of"
        failure formulation of the factor in the Bernoulli trial notation. The
        probability of failure is 1 - the contact weight.
    far_contact_weight : float
        Weight of contacts of type "far" (as compared to household)
        contacts. Similar to intermediate_contact_weight, this parameter is 
        formulated as a weight.
    age_transmission_discount : float
        Factor by which younger children are less likely to receive and transmit
        an infection. More specifically, the age_transmission_discount is the
        slope of a piecewise linear function that is 1 at age 18 (and above)
        and decreases for younger ages.
    prevention_measures : dictionary
        Dictionary listing all prevention measures in place for the given
        scenario. Fields that are not specifically included in this dictionary
        will revert to SEIRX_school defaults.
    school_characteristics: dictionary
        Dictionary holding the characteristics of each possible school type. 
        Needs to include the fields "classes" and "students" (i.e. the number)
        of students per class. The number of teachers is calculated
        automatically from the given school type and number of classes.
    agent_index_ratios : pandas DataFrame
        Data frame holding the empirically observed index case ratios for 
        students and teachers. Has to include the columns "school_type", 
        "student" and "teacher".
    simulation_params : dictionary
        Dictionary holding simulation parameters such as "verbosity" and
        "base_transmission_risk". Fields that are not included will revert back
        to SEIRX_school defaults.
    contact_network_src : string
        Absolute or relative path pointing to the location of the contact
        network used for the calibration runs. The location needs to hold the
        contact networks for each school types in a sub-folder with the same
        name as the school type. Networks need to be saved in networkx's .bz2
        format.
    N_steps : integer
        Number of maximum steps per run. This is a very conservatively chosen 
        value that ensures that an outbreak will always terminate within the 
        allotted time. Most runs are terminated way earlier anyways, as soon as 
        the outbreak is over.
        
    Returns
    -------
    model : SEIRX_school model instance holding a completed simulation run and
        all associated data.
    index_case : agent group from which the index case was drawn in the given
        simulation run.
    '''
    
    # since we only use contacts of type "close", "intermediate" and "far" in 
    # this setup, we set the contact type "very far" to 0. The contact type
    # "close" corresponds to household transmissions and is set to 1 (= base 
    # transmission risk). We therefore only calibrate the weight of the 
    # "intermediate"  and "far" contacts with respect to household contacts
    infection_risk_contact_type_weights = {
            'very_far': 0, 
            'far': far_contact_weight, 
            'intermediate': intermediate_contact_weight,
            'close': 1}
    
    # get the respective parameters for the given school type
    measures = prevention_measures[school_type]
    characteristics = school_characteristics[school_type]
    agent_index_ratio = agent_index_ratios.loc[school_type]
    # create the agent dictionaries based on the given parameter values and
    # prevention measures
    agent_types = compose_agents(measures)

    school_name = '{}_classes-{}_students-{}'.format(school_type,
                characteristics['classes'], characteristics['students'])
    school_src = join(contact_network_src, school_type)

    # load the contact graph: since households and sibling contacts
    # are random, there are a number of randomly created instances of 
    # calibration schools from which we can chose. We use a different
    # calibration school instance for every run here
    G = nx.readwrite.gpickle.read_gpickle(join(school_src,\
                    '{}_{}.bz2'.format(school_name, run)))

    # pick an index case according to the probabilities for the school type
    index_case = np.random.choice(['teacher', 'student'],
                                  p=[agent_index_ratios.loc['primary']['teacher'],
                                    agent_index_ratios.loc['primary']['student']])

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
                 infection_risk_contact_type_weights,
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
                 {'slope':age_transmission_discount, 'intercept':1},
      age_symptom_modification = simulation_params['age_symptom_discount'],
      mask_filter_efficiency = measures['mask_filter_efficiency'],
      transmission_risk_ventilation_modifier = \
                         measures['transmission_risk_ventilation_modifier'],)

    # run the model until the outbreak is over
    for i in range(N_steps):
        # break if first outbreak is over
        if len([a for a in model.schedule.agents if \
            (a.exposed == True or a.infectious == True)]) == 0:
            break
        model.step()
        
    return model, index_case


def run_ensemble(N_runs, school_type, intermediate_contact_weight,
              far_contact_weight, age_transmission_discount, 
              prevention_measures, school_characteristics,
              agent_index_ratios, simulation_params,
              contact_network_src, ensmbl_dst):
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
    intermediate_contact_weight : float
        Weight of contacts of type "intermediate" (as compared to household)
        contacts. Note: This parameter is formulated in terms of a "weight",
        i.e. a multiplicative factor to the intensity of the household contact
        (which is 1 by default). This is different from the "probability of"
        failure formulation of the factor in the Bernoulli trial notation. The
        probability of failure is 1 - the contact weight.
    far_contact_weight : float
        Weight of contacts of type "far" (as compared to household)
        contacts. Similar to intermediate_contact_weight, this parameter is 
        formulated as a weight.
    age_transmission_discount : float
        Factor by which younger children are less likely to receive and transmit
        an infection. More specifically, the age_transmission_discount is the
        slope of a piecewise linear function that is 1 at age 18 (and above)
        and decreases for younger ages.
    prevention_measures : dictionary
        Dictionary listing all prevention measures in place for the given
        scenario. Fields that are not specifically included in this dictionary
        will revert to SEIRX_school defaults.
    school_characteristics: dictionary
        Dictionary holding the characteristics of each possible school type. 
        Needs to include the fields "classes" and "students" (i.e. the number)
        of students per class. The number of teachers is calculated
        automatically from the given school type and number of classes.
    agent_index_ratios : pandas DataFrame
        Data frame holding the empirically observed index case ratios for 
        students and teachers. Has to include the columns "school_type", 
        "student" and "teacher".
    simulation_params : dictionary
        Dictionary holding simulation parameters such as "verbosity" and
        "base_transmission_risk". Fields that are not included will revert back
        to SEIRX_school defaults.
    contact_network_src : string
        Absolute or relative path pointing to the location of the contact
        network used for the calibration runs. The location needs to hold the
        contact networks for each school types in a sub-folder with the same
        name as the school type. Networks need to be saved in networkx's .bz2
        format.
    ensmbl_dst : string
        Absolute or relative path pointing to the location where full ensemble
        results should be saved. 
        
    Returns:
    --------
    ensemble_results : pandas DataFrame
        Data Frame holding the observable of interest of the ensemble, namely
        the number of infected students and teachers.
    '''
    
    ensemble_results = pd.DataFrame()
    ensemble_runs = pd.DataFrame()
    for run in range(1, N_runs + 1):
        model, index_case = run_model(school_type, run, 
                  intermediate_contact_weight,
                  far_contact_weight, age_transmission_discount, 
                  prevention_measures, school_characteristics,
                  agent_index_ratios, simulation_params,
                  contact_network_src)
        
        # collect the observables needed to calculate the difference to the
        # expected values
        infected_teachers = af.count_infected(model, 'teacher')
        infected_students = af.count_infected(model, 'student')
        # subtract the index case from the number of infected teachers/students
        # to arrive at the number of follow-up cases
        if index_case == 'teacher':
            infected_teachers -= 1
        else:
            infected_students -= 1

        # add run results to the ensemble results
        ensemble_results = ensemble_results.append({ 
                  'infected_teachers':infected_teachers,
                  'infected_students':infected_students}, ignore_index=True)
        
        # collect the statistics of the single run
        data = model.datacollector.get_model_vars_dataframe()
        data['run'] = run
        data['step'] = range(0, len(data))
        ensemble_runs = pd.concat([ensemble_runs, data])
        
    ensemble_runs = ensemble_runs.reset_index(drop=True)
    ensemble_runs.to_csv(join(ensmbl_dst,
            'school_type-{}_icw-{:1.2f}_fcw-{:1.2f}_atd-{:1.2f}.csv'\
        .format(school_type, intermediate_contact_weight, far_contact_weight,
                age_transmission_discount)), index=False)
        
    return ensemble_results   


def evaluate_ensemble(ensemble_results, school_type, intermediate_contact_weight,
                      far_contact_weight, age_transmission_discount,
                      outbreak_size, group_distribution):
    '''
    Utility function to calculate the error measures (chi-squared distance and
    sum of squared differences) between an ensemble of simulation runs for a
    given school type and parameter combination and the empirical outbreak size
    distribution and ratio of infected students vs. infected teachers.
    
    Parameters:
    -----------
    ensemble_results: pandas DataFrame
        Data Frame holding the observable of interest of the ensemble, namely
        the number of infected students and teachers.
    school_type : string
        School type for which the ensemble was run. Can be "primary", 
        "primary_dc", "lower_secondary", "lower_secondary_dc", 
        "upper_secondary", "secondary" or "secondary_dc".
    intermediate_contact_weight : float
        Weight of contacts of type "intermediate" (as compared to household)
        contacts. This parameter needs to be calibrated and is varied between
        ensembles. Note: This parameter is formulated in terms of a "weight",
        i.e. a multiplicative factor to the intensity of the household contact
        (which is 1 by default). This is different from the "probability of"
        failure formulation of the factor in the Bernoulli trial notation. The
        probability of failure is 1 - the contact weight.
    far_contact_weight : float
        Weight of contacts of type "far" (as compared to household)
        contacts. This parameter needs to be calibrated and is varied between
        ensembles. Similar to intermediate_contact_weight, this parameter is 
        formulated as a weight.
    age_transmission_discount : float
        Factor by which younger children are less likely to receive and transmit
        an infection. More specifically, the age_transmission_discount is the
        slope of a piecewise linear function that is 1 at age 18 (and above)
        and decreases for younger ages. This parameter needs to be calibrated 
        and is varied between ensembles.
    outbreak_size : pandas DataFrame
        Data frame holding the empirical outbreak size observations. The 
        outbreak size has to be given in the column "size", the school type in
        the column "type".
    group_distributions : pandas DataFrame
        Data frame holding the empirical observations of the ratio of infections
        in a given group (student, teacher) as compared to the overall number of
        infections (students + teachers). The data frame has three columns:
        "school_type", "group" and "ratio", where "group" indicates which group
        (student or teacher) the number in "ratio" belongs to.
        
    Returns:
    --------
    row : dictionary
        Dictionary holding the school type, values for the calibration 
        parameters (intermediate_contact_weight, far_contact_weight, 
        age_transmission_discount) and the values of the respective error terms
        for the outbreak size distribution and group case distribution.
    '''
    # calculate the differences between the expected and observed outbreak sizes
    # and the distribution of cases to the two agent groups
    chi2_distance_size, sum_of_squares_size = calculate_distribution_difference(\
                        school_type, ensemble_results, outbreak_size)
    chi2_distance_distro, sum_of_squares_distro = calculate_group_case_difference(\
                        school_type, ensemble_results, group_distribution)

    row = {
        'school_type':school_type,
        'intermediate_contact_weight':intermediate_contact_weight,
        'far_contact_weight':far_contact_weight,
        'age_transmission_discount':age_transmission_discount,
        'chi2_distance_size':chi2_distance_size,
        'sum_of_squares_size':sum_of_squares_size,
        'chi2_distance_distro':chi2_distance_distro,
        'sum_of_squares_distro':sum_of_squares_distro,
        'chi2_distance_total':chi2_distance_size + chi2_distance_distro,
        'sum_of_squares_total':sum_of_squares_size + sum_of_squares_distro
        }
    
    return row

def get_ensemble_parameters_from_filename(f):
    '''
    Extracts the simulation parameters for an ensemble given its ensemble file
    name string.
    
    Parameters:
    -----------
    f : string of the form school_type-{}_icw-{}_fcw-{}_atd-{}.csv that encodes
        the ensemble parameter for the school type, intermediate contact weight
        (icw), far contact weight (fcw) and age transmission discount (atd).
        The parameters icw, fcw and atd are floats with a precision of two 
        decimal places.
    
    Returns:
    --------
    params : dict
        Dict with the fields school_type (str), icw (float), fcw (float) and
        atd (float), which hold the simulation parameters of the ensemble.
    '''
    school_type = f.split('_icw')[0].replace('school_type-', '')
    icw = round(float(f.split('icw-')[1].split('_fcw')[0]), 2)
    fcw = round(float(f.split('fcw-')[1].split('_atd')[0]), 2)
    atd = round(float(f.split('atd-')[1].split('.csv')[0]), 2)
    params = {'school_type':school_type, 'icw':icw, 'fcw':fcw, 'atd':atd}
    
    return params

def evaulate_from_ensemble_data(src, ep, outbreak_sizes, group_distributions):
    '''
    Evaluate an ensemble from the simulation data saved for the ensemble 
    (rahter than the model object holding the finished simulation itself),
    given the expected outbreak sizes and group distributions.
    
    Parameters:
    -----------
    src : string
        Absolute or relative path to the folder holding all ensemble data.
    ep : tuple
        Tuple holding the ensemble parameters (number of runs, school type, 
        intermediate contact weight, far contact weight, age trans. discount).
    outbreak_size : pandas DataFrame
        Data frame holding the empirical outbreak size observations. The 
        outbreak size has to be given in the column "size", the school type in
        the column "type".
    group_distributions : pandas DataFrame
        Data frame holding the empirical observations of the ratio of infections
        in a given group (student, teacher) as compared to the overall number of
        infections (students + teachers). The data frame has three columns:
        "school_type", "group" and "ratio", where "group" indicates which group
        (student or teacher) the number in "ratio" belongs to.
        
    Returns:
    --------
    row : dictionary
        Dictionary holding the school type, values for the calibration 
        parameters (intermediate_contact_weight, far_contact_weight, 
        age_transmission_discount) and the values of the respective error terms
        for the outbreak size distribution and group case distribution.
    '''
    _, school_type, icw, fcw, atd = ep
    ensemble_results = pd.DataFrame()
    fname = 'school_type-{}_icw-{:1.2f}_fcw-{:1.2f}_atd-{:1.2f}.csv'\
        .format(school_type, icw, fcw, atd)
    ensmbl = pd.read_csv(join(src, fname))
    
    for run in ensmbl['run'].unique():
        run = ensmbl[ensmbl['run'] == run]
        
        # find the index case
        if (run[run['step'] == 0]['E_student'] == 1).values[0]:
            index_case = 'student'
        else:
            index_case = 'teacher'
            
        last_step = run[run['step'] == run['step'].max()]
        infected_students = last_step[['I_student', 'R_student', 'E_student']]\
            .sum(axis=1).values[0]
        infected_teachers = last_step[['I_teacher', 'R_teacher', 'E_teacher']]\
            .sum(axis=1).values[0]
        
        if index_case == 'teacher':
            infected_teachers -= 1
        else:
            infected_students -= 1
        
        # add run results to the ensemble results
        ensemble_results = ensemble_results.append({ 
                  'infected_teachers':infected_teachers,
                  'infected_students':infected_students}, ignore_index=True)
        
        row = cf.evaluate_ensemble(ensemble_results, school_type, icw,
                      fcw, atd, outbreak_sizes, group_distributions)
        
        return row