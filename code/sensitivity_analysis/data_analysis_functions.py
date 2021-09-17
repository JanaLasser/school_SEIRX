import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from os.path import join
import os
import matplotlib as mpl
from scseirx import analysis_functions as af
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

school_types = ['primary', 'primary_dc', 'lower_secondary', 
                'lower_secondary_dc', 'upper_secondary', 'secondary']


def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' 
    Creates and returns a color map that can be used in heat map figures. If 
    float_list is not provided, colour map graduates linearly between each color
    in hex_list. If float_list is provided, each color in hex_list is mapped to 
    the respective location in float_list. 
        
    Parameters
    ----------
    hex_list: list
        List of hex code strings
    float_list: list
        List of floats between 0 and 1, same length as hex_list. Must start with
        0 and end with 1.

    Returns
    ----------
    colour map
    '''
    
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side 
    from a prescribed midpoint value) e.g. im=ax1.imshow(array,
    norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100)).
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    
    
def get_data(stype, src_path):
    '''
    Convenience function to read all ensembles from different measures
    of a given school type and return one single data frame
    '''
    data = pd.DataFrame()
    stype_path = join(src_path, stype)
    files = os.listdir(stype_path)
    for f in files:
        params, agents, half = get_measures(f.strip('.csv'))

        ensmbl = pd.read_csv(join(stype_path, f))
        try:
            ensmbl = ensmbl.drop(columns=['Unnamed: 0'])
        except KeyError:
            pass

        ensmbl['preventive_test_type'] = params['preventive_test_type']
        ensmbl['index_case'] = params['index_case']
        ensmbl['transmission_risk_ventilation_modifier'] = \
            params['transmission_risk_ventilation_modifier']
        
        
        if ('class_size_reduction' in params.keys()) and not\
           ('half_classes' in params.keys()):
            if params['class_size_reduction'] > 0:
                params['half_classes'] = True
                ensmbl['half_classes'] = True
            else:
                params['half_classes'] = False
                ensmbl['half_classes'] = False
                
        if ('half_classes' in params.keys()) and not\
           ('class_size_reduction' in params.keys()):
            if params['half_classes']:
                params['class_size_reduction'] = 0.5
                ensmbl['class_size_reduction'] = 0.5
            else:
                params['class_size_reduction'] = 0.0
                ensmbl['class_size_reduction'] = 0.0
            
        ensmbl['half_classes'] = params['half_classes']
        ensmbl['class_size_reduction'] = params['class_size_reduction']
        ensmbl['student_testing_rate'] = params['student_test_rate']
        ensmbl['teacher_testing_rate'] = params['teacher_test_rate']
        ensmbl['mask_efficiency_inhale'] = params['mask_efficiency_inhale']
        ensmbl['mask_efficiency_exhale'] = params['mask_efficiency_exhale']
        ensmbl['base_transmission_risk_multiplier'] = \
            params['base_transmission_risk_multiplier']
        ensmbl['student_mask'] = agents['student']['mask']
        ensmbl['teacher_mask'] = agents['teacher']['mask']
        ensmbl['student_screening_interval'] = agents['student']\
            ['screening_interval']
        ensmbl['teacher_screening_interval'] = agents['teacher']\
            ['screening_interval']
        ensmbl['teacher_vaccination_ratio'] = agents['teacher']\
            ['vaccination_ratio']
        ensmbl['student_vaccination_ratio'] = agents['student']\
            ['vaccination_ratio']
        ensmbl['family_member_vaccination_ratio'] = agents['family_member']\
            ['vaccination_ratio']
        ensmbl['contact_weight'] = params['contact_weight']
        ensmbl['age_transmission_discount'] = params['age_transmission_discount']
        
        data = pd.concat([data, ensmbl])

    data = data.reset_index(drop=True)
    data['teacher_screening_interval'] = data['teacher_screening_interval']\
        .replace({None:'never'})
    data['student_screening_interval'] = data['student_screening_interval']\
        .replace({None:'never'})
    return data


def get_measures(measure_string):
    '''
    Convenience function to get the individual measures given a string 
    (filename) of measures.
    '''
    agents = {
        'student':{
                'screening_interval': None, 
                'index_probability': 0, 
                'mask':False},
        'teacher':{
                'screening_interval': None, 
                'index_probability': 0, 
                'mask':False},
        'family_member':{
                'screening_interval': None, 
                'index_probability': 0, 
                'mask':False} 
}
    
    turnovers = {0:'same', 1:'one', 2:'two', 3:'three'}
    bmap = {'T':True, 'F':False}
    interval_map = {'0':0, '3':3, '7':7, '14':14, 'None':None}
    index_map = {'s':'student', 't':'teacher'}
    
    stype, _ = measure_string.split('_test')
    rest = measure_string.split(stype + '_')[1]
    
    ttpype, turnover, index, tf, sf, tmask, smask, vent, stestrate, \
    ttestrate, trisk, meffexh, meffinh, csizered, fratio, svacc, tvacc, \
    fvacc, atd, cw = rest.split('_')

    tmp = [ttpype, turnover, index, tf, sf, tmask, smask, vent, stestrate,\
       ttestrate, trisk, meffexh, meffinh, csizered, fratio, svacc, tvacc,\
       fvacc, atd, cw]
        

    tmp = [m.split('-') for m in tmp]
    params = {}
    
    half = False
    for m in tmp:
        if len(m) == 1:
            pass
        elif m[0] == 'test':
            params['preventive_test_type'] = m[1]
        elif m[0] == 'turnover':
            params['turnover'] = int(m[1])
        elif m[0] == 'index':
            params['index_case'] = index_map[m[1]]
        elif m[0] == 'tf':
            agents['teacher']['screening_interval'] = interval_map[m[1]]
        elif m[0] == 'sf':
            agents['student']['screening_interval'] = interval_map[m[1]]
        elif m[0] == 'tmask':
            agents['teacher']['mask'] = bmap[m[1]]    
        elif m[0] == 'smask':
            agents['student']['mask'] = bmap[m[1]]
        elif m[0] == 'half':
            params['half_classes'] = bmap[m[1]]
        elif m[0] == 'vent':
            params['transmission_risk_ventilation_modifier'] = float(m[1])
        elif m[0] == 'csizered':
            params['class_size_reduction'] = float(m[1])
        elif m[0] == 'stestrate':
            params['student_test_rate'] = float(m[1])
        elif m[0] == 'ttestrate':
            params['teacher_test_rate'] = float(m[1])
        elif m[0] == 'fratio':
            params['added_friendship_contacts'] = float(m[1])
        elif m[0] == 'meffexh':
            params['mask_efficiency_exhale'] = float(m[1])
        elif m[0] == 'meffinh':
            params['mask_efficiency_inhale'] = float(m[1])
        elif m[0] == 'trisk':
            params['base_transmission_risk_multiplier'] = float(m[1])
        elif m[0] == 'tvacc':
            agents['teacher']['vaccination_ratio'] = float(m[1])
        elif m[0] == 'svacc':
            agents['student']['vaccination_ratio'] = float(m[1])
        elif m[0] == 'fvacc':
            agents['family_member']['vaccination_ratio'] = float(m[1])
        elif m[0] == 'atd':
            params['age_transmission_discount'] = -float(m[2])
        elif m[0] == 'cw':
            params['contact_weight'] = float(m[1])
        else:
            print('unknown measure type ', m[0])
            
    params['preventive_test_type'] = '{}_day_{}'\
    .format(turnovers[params['turnover']], params['preventive_test_type'])
    
    return params, agents, half


def get_baseline_data(src_path,
        school_types=['primary', 'primary_dc', 'lower_secondary',
                      'lower_secondary_dc', 'upper_secondary', 'secondary']):
    baseline_data = pd.DataFrame()

    for stype in school_types:
        tmp = pd.read_csv(join(src_path, '{}_observables.csv'.format(stype)))
        tmp['school_type'] = stype
        baseline_data = pd.concat([baseline_data, tmp])


    baseline_data['test_sensitivity'] = 1.0
    baseline_data['student_testing_rate'] = 1.0
    baseline_data['teacher_testing_rate'] = 1.0
    baseline_data['mask_efficiency_inhale'] = 0.7
    baseline_data['mask_efficiency_exhale'] = 0.5
    baseline_data['base_transmission_risk_multiplier'] = 1.0
    baseline_data['friendship_ratio'] = 0.0
    baseline_data['student_vaccination_ratio'] = 0.0
    baseline_data['teacher_vaccination_ratio'] = 0.0
    baseline_data['family_member_vaccination_ratio'] = 0.0
    baseline_data['class_size_reduction'] = 0
    baseline_data.loc[baseline_data[baseline_data['half_classes'] == True].index,
                      'class_size_reduction'] = 0.5

    baseline_data = baseline_data.drop(columns=['Unnamed: 0'])
    baseline_data = baseline_data.reset_index(drop=True)
    baseline_data['student_screen_interval'] = \
        baseline_data['student_screen_interval'].replace({np.nan:'never'})
    baseline_data['teacher_screen_interval'] = \
        baseline_data['teacher_screen_interval'].replace({np.nan:'never'})
    
    return baseline_data


def get_test_sensitivity_data(src_path, params, baseline_data, 
        school_types=['primary', 'primary_dc', 'lower_secondary',
                      'lower_secondary_dc', 'upper_secondary', 'secondary'],
        observables_of_interest=['infected_agents', 'R0']):
    
    test_sensitivity_data = pd.DataFrame()
    for stype in school_types:
        print('\t{}'.format(stype))
        stype_data = get_data(stype, src_path)

        for i, screening_params in params.iterrows():
            index_case, s_screen_interval, t_screen_interval, student_mask, \
            teacher_mask, class_size_reduction, vent_mod, ttype = \
                screening_params

            turnover, _, test = ttype.split('_')
            sensitivity = float(test.split('antigen')[1])

            # calculate the ensemble statistics for each measure combination 
            measure_data = stype_data[\
                (stype_data['preventive_test_type'] == ttype) &\
                (stype_data['index_case'] == index_case) &\
                (stype_data['student_screening_interval'] == s_screen_interval) &\
                (stype_data['teacher_screening_interval'] == t_screen_interval) &\
                (stype_data['student_mask'] == student_mask) &\
                (stype_data['teacher_mask'] == teacher_mask) &\
                (stype_data['class_size_reduction'] == class_size_reduction) &\
                (stype_data['transmission_risk_ventilation_modifier'] == vent_mod)]

            if len(measure_data) == 0:
                print('WARNING: empty measure data for {}'\
                      .format(screening_params))

            row = {'school_type':stype,
                   'test_type':test,
                   'turnover':turnover,
                   'index_case':index_case,
                   'student_screen_interval':s_screen_interval,
                   'teacher_screen_interval':t_screen_interval,
                   'student_mask':student_mask,
                   'teacher_mask':teacher_mask,
                   'ventilation_modification':vent_mod,
                   'test_sensitivity':sensitivity,
                   'class_size_reduction':class_size_reduction,
                   'student_testing_rate':1.0,
                   'teacher_testing_rate':1.0,
                   'mask_efficiency_inhale':0.7,
                   'mask_efficiency_exhale':0.5,
                   'base_transmission_risk_multiplier':1.0,
                   'friendship_ratio':0,
                   'student_vaccination_ratio':0,
                   'teacher_vaccination_ratio':0,
                   'family_member_vaccination_ratio':0}

            for col in observables_of_interest:
                row.update(af.get_statistics(measure_data, col))

            test_sensitivity_data = test_sensitivity_data.append(row,
                                                            ignore_index=True)

    test_sensitivity_data.to_csv(join(src_path, 'test_sensitivity_observables.csv'),
                               index=False)        

    # combine the sensitivity analysis data with the baseline data 
    # (only relevant columns)
    baseline_chunk = baseline_data[\
                (baseline_data['test_type'] == 'antigen') &\
                (baseline_data['turnover'] == 0) &\
                (baseline_data['student_screen_interval'] == s_screen_interval) &\
                (baseline_data['teacher_screen_interval'] == t_screen_interval) &\
                (baseline_data['student_mask'] == student_mask) &\
                (baseline_data['teacher_mask'] == teacher_mask) &\
                (baseline_data['class_size_reduction'] == class_size_reduction) &\
                (baseline_data['ventilation_modification'] == vent_mod)]
    
    test_sensitivity_data = pd.concat([test_sensitivity_data, \
                            baseline_chunk[test_sensitivity_data.columns].copy()])
    
    return test_sensitivity_data


def get_testing_rate_data(src_path, params, baseline_data, 
        school_types=['primary', 'primary_dc', 'lower_secondary',
                      'lower_secondary_dc', 'upper_secondary', 'secondary'],
        observables_of_interest=['infected_agents', 'R0']):
    
    testing_rate_data = pd.DataFrame()
    for stype in school_types:
        print('\t{}'.format(stype))
        stype_data = get_data(stype, src_path)

        for i, screening_params in params.iterrows():
            index_case, s_screen_interval, t_screen_interval, student_mask, \
            teacher_mask, class_size_reduction, vent_mod, s_testing_rate, \
            t_testing_rate = screening_params

            measure_data = stype_data[\
                (stype_data['preventive_test_type'] == 'same_day_antigen') &\
                (stype_data['index_case'] == index_case) &\
                (stype_data['student_screening_interval'] == s_screen_interval) &\
                (stype_data['teacher_screening_interval'] == t_screen_interval) &\
                (stype_data['student_mask'] == student_mask) &\
                (stype_data['teacher_mask'] == teacher_mask) &\
                (stype_data['class_size_reduction'] == class_size_reduction) &\
                (stype_data['transmission_risk_ventilation_modifier'] == vent_mod) &\
                (stype_data['student_testing_rate'] == s_testing_rate) &\
                (stype_data['teacher_testing_rate'] == t_testing_rate)
                                     ]

            if len(measure_data) == 0:
                print('WARNING: empty measure data for {}'.format(screening_params))

            row = {'school_type':stype,
                   'test_type':'antigen',
                   'turnover':0,
                   'index_case':index_case,
                   'student_screen_interval':s_screen_interval,
                   'teacher_screen_interval':t_screen_interval,
                   'student_mask':student_mask,
                   'teacher_mask':teacher_mask,
                   'ventilation_modification':vent_mod,
                   'test_sensitivity':1.0,
                   'class_size_reduction':class_size_reduction,
                   'student_testing_rate':s_testing_rate,
                   'teacher_testing_rate':t_testing_rate,
                   'mask_efficiency_inhale':0.7,
                   'mask_efficiency_exhale':0.5,
                   'base_transmission_risk_multiplier':1.0,
                   'friendship_ratio':0,
                   'student_vaccination_ratio':0,
                   'teacher_vaccination_ratio':0,
                   'family_member_vaccination_ratio':0}

            for col in observables_of_interest:
                row.update(af.get_statistics(measure_data, col))

            testing_rate_data = testing_rate_data.append(row, ignore_index=True)

    testing_rate_data.to_csv(join(src_path, 'testing_rate_data_observables.csv'),
                             index=False)

    # combine the sensitivity analysis data with the baseline data 
    # (only relevant columns)
    baseline_chunk = baseline_data[\
                (baseline_data['test_type'] == 'antigen') &\
                (baseline_data['turnover'] == 0) &\
                (baseline_data['student_screen_interval'] == s_screen_interval) &\
                (baseline_data['teacher_screen_interval'] == t_screen_interval) &\
                (baseline_data['student_mask'] == student_mask) &\
                (baseline_data['teacher_mask'] == teacher_mask) &\
                (baseline_data['class_size_reduction'] == class_size_reduction) &\
                (baseline_data['ventilation_modification'] == vent_mod)]
    testing_rate_data = pd.concat([testing_rate_data, \
                        baseline_chunk[testing_rate_data.columns].copy()])
    
    return testing_rate_data


def get_class_size_reduction_data(src_path, params, 
        school_types=['primary', 'primary_dc', 'lower_secondary',
                      'lower_secondary_dc', 'upper_secondary', 'secondary'],
        observables_of_interest=['infected_agents', 'R0']):
    
    class_size_reduction_data = pd.DataFrame()
    
    for stype in school_types:
        print('\t{}'.format(stype))
        stype_data = get_data(stype, src_path)

        for i, screening_params in params.iterrows():
            index_case, s_screen_interval, t_screen_interval, student_mask, \
            teacher_mask, vent_mod, class_size_reduction = screening_params

            # calculate the ensemble statistics for each measure combination 
            measure_data = stype_data[\
                (stype_data['preventive_test_type'] == 'same_day_antigen') &\
                (stype_data['index_case'] == index_case) &\
                (stype_data['student_screening_interval'] == s_screen_interval) &\
                (stype_data['teacher_screening_interval'] == t_screen_interval) &\
                (stype_data['student_mask'] == student_mask) &\
                (stype_data['teacher_mask'] == teacher_mask) &\
                (stype_data['class_size_reduction'] == class_size_reduction) &\
                (stype_data['transmission_risk_ventilation_modifier'] == vent_mod)
                                     ]

            if len(measure_data) == 0:
                print('WARNING: empty measure data for {}'.format(screening_params))

            row = {'school_type':stype,
                   'test_type':'antigen',
                   'turnover':0,
                   'index_case':index_case,
                   'student_screen_interval':s_screen_interval,
                   'teacher_screen_interval':t_screen_interval,
                   'student_mask':student_mask,
                   'teacher_mask':teacher_mask,
                   'ventilation_modification':vent_mod,
                   'test_sensitivity':1.0,
                   'class_size_reduction':class_size_reduction,
                   'student_testing_rate':1.0,
                   'teacher_testing_rate':1.0,
                   'mask_efficiency_inhale':0.7,
                   'mask_efficiency_exhale':0.5,
                   'base_transmission_risk_multiplier':1.0,
                   'friendship_ratio':0,
                   'student_vaccination_ratio':0,
                   'teacher_vaccination_ratio':0,
                   'family_member_vaccination_ratio':0}

            for col in observables_of_interest:
                row.update(af.get_statistics(measure_data, col))

            class_size_reduction_data = \
                    class_size_reduction_data.append(row, ignore_index=True)

    class_size_reduction_data.to_csv(join(src_path.split('/ensembles')[0],
                            'class_size_reduction_observables.csv'), index=False)
    return class_size_reduction_data
    
    
def get_ventilation_efficiency_data(src_path, params, baseline_data,
        school_types=['primary', 'primary_dc', 'lower_secondary',
                      'lower_secondary_dc', 'upper_secondary', 'secondary'],
        observables_of_interest=['infected_agents', 'R0']):
    
    ventilation_efficiency_data = pd.DataFrame()
    
    for stype in school_types:
        print('\t{}'.format(stype))
        stype_data = get_data(stype, src_path)

        for i, screening_params in params.iterrows():
            index_case, s_screen_interval, t_screen_interval, student_mask, \
            teacher_mask, class_size_reduction, vent_mod = screening_params

            # calculate the ensemble statistics for each measure combination 
            measure_data = stype_data[\
                (stype_data['preventive_test_type'] == 'same_day_antigen') &\
                (stype_data['index_case'] == index_case) &\
                (stype_data['student_screening_interval'] == s_screen_interval) &\
                (stype_data['teacher_screening_interval'] == t_screen_interval) &\
                (stype_data['student_mask'] == student_mask) &\
                (stype_data['teacher_mask'] == teacher_mask) &\
                (stype_data['class_size_reduction'] == class_size_reduction) &\
                (stype_data['transmission_risk_ventilation_modifier'] == vent_mod)
                                     ]

            if len(measure_data) == 0:
                print('WARNING: empty measure data for {}'.format(screening_params))

            row = {'school_type':stype,
                   'test_type':'antigen',
                   'turnover':0,
                   'index_case':index_case,
                   'student_screen_interval':s_screen_interval,
                   'teacher_screen_interval':t_screen_interval,
                   'student_mask':student_mask,
                   'teacher_mask':teacher_mask,
                   'ventilation_modification':vent_mod,
                   'test_sensitivity':1.0,
                   'class_size_reduction':class_size_reduction,
                   'student_testing_rate':1.0,
                   'teacher_testing_rate':1.0,
                   'mask_efficiency_inhale':0.7,
                   'mask_efficiency_exhale':0.5,
                   'base_transmission_risk_multiplier':1.0,
                   'friendship_ratio':0,
                   'student_vaccination_ratio':0,
                   'teacher_vaccination_ratio':0,
                   'family_member_vaccination_ratio':0}

            for col in observables_of_interest:
                row.update(af.get_statistics(measure_data, col))

            ventilation_efficiency_data = \
                    ventilation_efficiency_data.append(row, ignore_index=True)

    ventilation_efficiency_data.to_csv(join(src_path.split('/ensembles')[0],
                            'ventilation_efficiency_observables.csv'), index=False)    

    # combine the sensitivity analysis data with the baseline data 
    # (only relevant columns)
    baseline_chunk = baseline_data[\
                (baseline_data['test_type'] == 'antigen') &\
                (baseline_data['turnover'] == 0) &\
                (baseline_data['student_screen_interval'] == s_screen_interval) &\
                (baseline_data['teacher_screen_interval'] == t_screen_interval) &\
                (baseline_data['student_mask'] == student_mask) &\
                (baseline_data['teacher_mask'] == teacher_mask) &\
                (baseline_data['class_size_reduction'] == class_size_reduction) &\
                (baseline_data['ventilation_modification'] == 0.36)]
    ventilation_efficiency_data = pd.concat([ventilation_efficiency_data, \
                    baseline_chunk[ventilation_efficiency_data.columns].copy()])
    
    return ventilation_efficiency_data


def get_mask_efficiency_data(src_path, params,
        school_types=['primary', 'primary_dc', 'lower_secondary',
                      'lower_secondary_dc', 'upper_secondary', 'secondary'],
        observables_of_interest=['infected_agents', 'R0']):
    
    mask_efficiency_data = pd.DataFrame()
    
    for stype in school_types:
        print('\t{}'.format(stype))
        stype_data = get_data(stype, src_path)

        for i, screening_params in params.iterrows():
            index_case, s_screen_interval, t_screen_interval, student_mask, \
            teacher_mask, class_size_reduction, vent_mod, m_efficiency_exhale, \
            m_efficiency_inhale = screening_params

            # calculate the ensemble statistics for each measure combination 
            measure_data = stype_data[\
                (stype_data['preventive_test_type'] == 'same_day_antigen') &\
                (stype_data['index_case'] == index_case) &\
                (stype_data['student_screening_interval'] == s_screen_interval) &\
                (stype_data['teacher_screening_interval'] == t_screen_interval) &\
                (stype_data['student_mask'] == student_mask) &\
                (stype_data['teacher_mask'] == teacher_mask) &\
                (stype_data['class_size_reduction'] == class_size_reduction) &\
                (stype_data['transmission_risk_ventilation_modifier'] == vent_mod) &\
                (stype_data['mask_efficiency_inhale'] == m_efficiency_inhale) &\
                (stype_data['mask_efficiency_exhale'] == m_efficiency_exhale)
                                     ]

            if len(measure_data) == 0:
                print('WARNING: empty measure data for {}'.format(screening_params))

            row = {'school_type':stype,
                   'test_type':'antigen',
                   'turnover':0,
                   'index_case':index_case,
                   'student_screen_interval':s_screen_interval,
                   'teacher_screen_interval':t_screen_interval,
                   'student_mask':student_mask,
                   'teacher_mask':teacher_mask,
                   'ventilation_modification':vent_mod,
                   'test_sensitivity':1.0,
                   'class_size_reduction':0.0,
                   'student_testing_rate':1.0,
                   'teacher_testing_rate':1.0,
                   'mask_efficiency_inhale':m_efficiency_exhale,
                   'mask_efficiency_exhale':m_efficiency_inhale,
                   'base_transmission_risk_multiplier':1.0,
                   'friendship_ratio':0,
                   'student_vaccination_ratio':0,
                   'teacher_vaccination_ratio':0,
                   'family_member_vaccination_ratio':0}

            for col in observables_of_interest:
                row.update(af.get_statistics(measure_data, col))

            mask_efficiency_data = \
                    mask_efficiency_data.append(row, ignore_index=True)

    mask_efficiency_data.to_csv(join(src_path.split('/ensembles')[0],
                            'mask_efficiency_observables.csv'), index=False)  
    
    return mask_efficiency_data


def get_added_friendship_contacts_data(src_path, params, baseline_data,
        school_types=['primary', 'primary_dc', 'lower_secondary',
                      'lower_secondary_dc', 'upper_secondary', 'secondary'],
        observables_of_interest=['infected_agents', 'R0']):
    
    added_friendship_contacts_data = pd.DataFrame()
    
    for stype in school_types:
        print('\t{}'.format(stype))
        stype_data = get_data(stype, src_path)

        for i, screening_params in params.iterrows():
            index_case, s_screen_interval, t_screen_interval, student_mask, \
            teacher_mask, class_size_reduction, vent_mod, friendship_ratio = screening_params

            # calculate the ensemble statistics for each measure combination 
            measure_data = stype_data[\
                (stype_data['preventive_test_type'] == 'same_day_antigen') &\
                (stype_data['index_case'] == index_case) &\
                (stype_data['student_screening_interval'] == s_screen_interval) &\
                (stype_data['teacher_screening_interval'] == t_screen_interval) &\
                (stype_data['student_mask'] == student_mask) &\
                (stype_data['teacher_mask'] == teacher_mask) &\
                (stype_data['class_size_reduction'] == class_size_reduction) &\
                (stype_data['transmission_risk_ventilation_modifier'] == vent_mod) &\
                (stype_data['friendship_ratio'] == friendship_ratio)
                                     ]

            if len(measure_data) == 0:
                print('WARNING: empty measure data for {}'.format(screening_params))

            row = {'school_type':stype,
                   'test_type':'antigen',
                   'turnover':0,
                   'index_case':index_case,
                   'student_screen_interval':s_screen_interval,
                   'teacher_screen_interval':t_screen_interval,
                   'student_mask':student_mask,
                   'teacher_mask':teacher_mask,
                   'ventilation_modification':vent_mod,
                   'test_sensitivity':1.0,
                   'class_size_reduction':class_size_reduction,
                   'student_testing_rate':1.0,
                   'teacher_testing_rate':1.0,
                   'mask_efficiency_inhale':0.7,
                   'mask_efficiency_exhale':0.5,
                   'base_transmission_risk_multiplier':1.0,
                   'friendship_ratio':friendship_ratio,
                   'student_vaccination_ratio':0,
                   'teacher_vaccination_ratio':0,
                   'family_member_vaccination_ratio':0}

            for col in observables_of_interest:
                row.update(af.get_statistics(measure_data, col))

            added_friendship_contacts_data = \
                    added_friendship_contacts_data.append(row, ignore_index=True)

    added_friendship_contacts_data.to_csv(join(src_path.split('/ensembles')[0],
                            'added_friendship_contacts_observables.csv'), index=False)      

    baseline_chunk = baseline_data[\
                (baseline_data['test_type'] == 'antigen') &\
                (baseline_data['turnover'] == 0) &\
                (baseline_data['student_screen_interval'] == 'never') &\
                (baseline_data['teacher_screen_interval'] == 'never') &\
                (baseline_data['student_mask'] == False) &\
                (baseline_data['teacher_mask'] == False) &\
                (baseline_data['class_size_reduction'] == 0.0) &\
                (baseline_data['ventilation_modification'] == 1.0) &\
                (baseline_data['friendship_ratio'] == 0.0)]
    
    # TODO dirty hack. fix this and figure out why there are duplicates
    baseline_chunk = baseline_chunk.drop_duplicates()
    added_friendship_contacts_data = pd.concat([added_friendship_contacts_data, \
                    baseline_chunk[added_friendship_contacts_data.columns].copy()])
    
    return added_friendship_contacts_data


def get_worst_case_data(src_path, params,
        school_types=['primary', 'primary_dc', 'lower_secondary',
                      'lower_secondary_dc', 'upper_secondary', 'secondary'],
        observables_of_interest=['infected_agents', 'R0']):
    
    worst_case_data = pd.DataFrame()
    for stype in school_types:
        stype_data = get_data(stype, src_path)

        for i, screening_params in params.iterrows():
            index_case, s_screen_interval, t_screen_interval, student_mask, \
            teacher_mask, class_size_reduction, vent_mod, m_efficiency_exhale, \
            m_efficiency_inhale, s_test_rate, t_test_rate, ttype, friendship_ratio \
                = screening_params

            turnover, _, test = ttype.split('_')
            sensitivity = float(test.split('antigen')[1]) 

            # calculate the ensemble statistics for each measure combination 
            measure_data = stype_data[\
                (stype_data['preventive_test_type'] == ttype) &\
                (stype_data['index_case'] == index_case) &\
                (stype_data['student_screening_interval'] == s_screen_interval) &\
                (stype_data['teacher_screening_interval'] == t_screen_interval) &\
                (stype_data['student_mask'] == student_mask) &\
                (stype_data['teacher_mask'] == teacher_mask) &\
                (stype_data['class_size_reduction'] == class_size_reduction) &\
                (stype_data['transmission_risk_ventilation_modifier'] == vent_mod)]

            if len(measure_data) == 0:
                print('WARNING: empty measure data for {}'.format(screening_params))
                
            half = False
            if class_size_reduction > 0:
                half = True

            row = {'school_type':stype,
                   'test_type':test,
                   'turnover':turnover,
                   'index_case':index_case,
                   'student_screen_interval':s_screen_interval,
                   'teacher_screen_interval':t_screen_interval,
                   'student_mask':student_mask,
                   'teacher_mask':teacher_mask,
                   'ventilation_modification':vent_mod,
                   'test_sensitivity':sensitivity,
                   'class_size_reduction':class_size_reduction,
                   'half_classes':half,
                   'student_testing_rate':s_test_rate,
                   'teacher_testing_rate':t_test_rate,
                   'mask_efficiency_inhale':m_efficiency_inhale,
                   'mask_efficiency_exhale':m_efficiency_exhale,
                   'base_transmission_risk_multiplier':1.0,
                   'friendship_ratio':friendship_ratio,
                   'student_vaccination_ratio':0,
                   'teacher_vaccination_ratio':0,
                   'family_member_vaccination_ratio':0}

            for col in observables_of_interest:
                row.update(af.get_statistics(measure_data, col))

            worst_case_data = \
                    worst_case_data.append(row, ignore_index=True)
            
    worst_case_data['scenario'] = 'conservative'
    return worst_case_data


def get_worst_case_and_vaccinations_data(src_path, params,
        school_types=['primary', 'primary_dc', 'lower_secondary',
                      'lower_secondary_dc', 'upper_secondary', 'secondary'],
        observables_of_interest=['infected_agents', 'R0']):
    
    worst_case_and_vaccinations_data = pd.DataFrame()
    for stype in school_types:
        print('\t{}'.format(stype))
        stype_data = get_data(stype, src_path)

        for i, screening_params in params.iterrows():
            index_case, s_screen_interval, t_screen_interval, student_mask, \
            teacher_mask, class_size_reduction, vent_mod, m_efficiency_exhale, \
            m_efficiency_inhale, s_test_rate, t_test_rate, ttype, \
            friendship_ratio, student_vaccination_ratio, \
            teacher_vaccination_ratio, family_member_vaccination_ratio \
            = screening_params

            turnover, _, test = ttype.split('_')
            sensitivity = float(test.split('antigen')[1]) 

            # calculate the ensemble statistics for each measure combination 
            measure_data = stype_data[\
                (stype_data['preventive_test_type'] == ttype) &\
                (stype_data['index_case'] == index_case) &\
                (stype_data['student_screening_interval'] \
                     == s_screen_interval) &\
                (stype_data['teacher_screening_interval'] \
                     == t_screen_interval) &\
                (stype_data['student_mask'] == student_mask) &\
                (stype_data['teacher_mask'] == teacher_mask) &\
                (stype_data['class_size_reduction'] == class_size_reduction) &\
                (stype_data['transmission_risk_ventilation_modifier'] \
                     == vent_mod) &\
                (stype_data['student_vaccination_ratio']\
                     == student_vaccination_ratio) &\
                (stype_data['teacher_vaccination_ratio']\
                     == teacher_vaccination_ratio) &\
                (stype_data['family_member_vaccination_ratio']\
                     == family_member_vaccination_ratio)]

            if len(measure_data) == 0:
                print('WARNING: empty measure data for {}'.format(screening_params))
                
            half = False
            if class_size_reduction > 0:
                half = True

            row = {'school_type':stype,
                   'test_type':test,
                   'turnover':turnover,
                   'index_case':index_case,
                   'student_screen_interval':s_screen_interval,
                   'teacher_screen_interval':t_screen_interval,
                   'student_mask':student_mask,
                   'teacher_mask':teacher_mask,
                   'ventilation_modification':vent_mod,
                   'test_sensitivity':sensitivity,
                   'class_size_reduction':class_size_reduction,
                   'half_classes':half,
                   'student_testing_rate':s_test_rate,
                   'teacher_testing_rate':t_test_rate,
                   'mask_efficiency_inhale':m_efficiency_inhale,
                   'mask_efficiency_exhale':m_efficiency_exhale,
                   'base_transmission_risk_multiplier':1.0,
                   'friendship_ratio':friendship_ratio,
                   'student_vaccination_ratio':student_vaccination_ratio,
                   'teacher_vaccination_ratio':teacher_vaccination_ratio,
                   'family_member_vaccination_ratio':\
                          family_member_vaccination_ratio}

            for col in observables_of_interest:
                row.update(af.get_statistics(measure_data, col))

            worst_case_and_vaccinations_data = \
                    worst_case_and_vaccinations_data.append(row, ignore_index=True)
            
    return worst_case_and_vaccinations_data


def get_vaccination_data(src_path, params,
        school_types=['primary', 'primary_dc', 'lower_secondary',
                      'lower_secondary_dc', 'upper_secondary', 'secondary'],
        observables_of_interest=['infected_agents', 'R0']):
    
    vaccination_data = pd.DataFrame()
    for stype in school_types:
        print('\t{}'.format(stype))
        stype_data = get_data(stype, src_path, vaccinations=True)

        for i, screening_params in params.iterrows():
            index_case, s_screen_interval, t_screen_interval, student_mask, \
            teacher_mask, half, vent_mod, student_vaccination_ratio, \
            teacher_vaccination_ratio, family_member_vaccination_ratio \
            = screening_params

            test = 'antigen'
            turnover = 0
            sensitivity = 1.0

            # calculate the ensemble statistics for each measure combination 
            measure_data = stype_data[\
                (stype_data['preventive_test_type'] == 'same_day_antigen') &\
                (stype_data['index_case'] == index_case) &\
                (stype_data['student_screening_interval'] \
                     == s_screen_interval) &\
                (stype_data['teacher_screening_interval'] \
                     == t_screen_interval) &\
                (stype_data['student_mask'] == student_mask) &\
                (stype_data['teacher_mask'] == teacher_mask) &\
                (stype_data['half_classes'] == half) &\
                (stype_data['transmission_risk_ventilation_modifier'] == \
                     vent_mod) &\
                (stype_data['student_vaccination_ratio'] == \
                     student_vaccination_ratio) &\
                (stype_data['teacher_vaccination_ratio'] == \
                     teacher_vaccination_ratio) &\
                (stype_data['family_member_vaccination_ratio'] == \
                     family_member_vaccination_ratio)]

            if len(measure_data) == 0:
                print('WARNING: empty measure data for {}'.format(screening_params))
                
            class_size_reduction = 0.0
            if half:
                class_size_reduction = 0.5

            row = {'school_type':stype,
                   'test_type':test,
                   'turnover':turnover,
                   'index_case':index_case,
                   'student_screen_interval':s_screen_interval,
                   'teacher_screen_interval':t_screen_interval,
                   'student_mask':student_mask,
                   'teacher_mask':teacher_mask,
                   'ventilation_modification':vent_mod,
                   'test_sensitivity':sensitivity,
                   'class_size_reduction':class_size_reduction,
                   'half_classes':half,
                   'student_testing_rate':1.0,
                   'teacher_testing_rate':1.0,
                   'mask_efficiency_inhale':0.5,
                   'mask_efficiency_exhale':0.7,
                   'base_transmission_risk_multiplier':1.0,
                   'friendship_ratio':0.0,
                   'student_vaccination_ratio':student_vaccination_ratio,
                   'teacher_vaccination_ratio':teacher_vaccination_ratio,
                   'family_member_vaccination_ratio':\
                       family_member_vaccination_ratio}

            for col in observables_of_interest:
                row.update(af.get_statistics(measure_data, col))

            vaccination_data = \
                    vaccination_data.append(row, ignore_index=True)
            
    return vaccination_data


def build_test_sensitivity_heatmap(test_sensitivity_data, sensitivities, 
                                   school_types=school_types):

    data = test_sensitivity_data[\
            (test_sensitivity_data['student_screen_interval'] == 7) &\
            (test_sensitivity_data['teacher_screen_interval'] == 7) &\
            (test_sensitivity_data['student_mask'] == False) &\
            (test_sensitivity_data['teacher_mask'] == False) &\
            (test_sensitivity_data['class_size_reduction'] == 0.0) &\
            (test_sensitivity_data['ventilation_modification'] == 1.0)]
    data = data.set_index(\
                ['school_type', 'test_sensitivity', 'index_case'])

    hmaps_test_sensitivity = {'N_infected':{'student':np.nan, 'teacher':np.nan},
                 'R0':{'student':np.nan, 'teacher':np.nan}}

    for index_case in ['student', 'teacher']:
        cmap = np.zeros((len(sensitivities), len(school_types)))
        cmap_R0 = np.zeros((len(sensitivities), len(school_types)))
        for i, s in enumerate(sensitivities):
            for j, st in enumerate(school_types):
                bl_infected_agents = data.loc[st, 1.0, index_case]\
                    ['infected_agents_mean']
                bl_R0 = data.loc[st, 1.0, index_case]['R0_mean']
                cmap[i, j] = data.loc[st, s, index_case]['infected_agents_mean'] / \
                    bl_infected_agents
                cmap_R0[i, j] = data .loc[st, s, index_case]['R0_mean']

        hmaps_test_sensitivity['N_infected'][index_case] = cmap
        hmaps_test_sensitivity['R0'][index_case] = cmap_R0
        
    return hmaps_test_sensitivity


def build_testing_rate_heatmaps(testing_rate_data, testing_rates,
                                school_types=school_types):
    
    data = testing_rate_data[\
            (testing_rate_data['student_screen_interval'] == 7) &\
            (testing_rate_data['teacher_screen_interval'] == 7) &\
            (testing_rate_data['student_mask'] == False) &\
            (testing_rate_data['teacher_mask'] == False) &\
            (testing_rate_data['class_size_reduction'] == 0.0) &\
            (testing_rate_data['ventilation_modification'] == 1.0)]

    data = data.set_index(\
                ['school_type', 'student_testing_rate', 'teacher_testing_rate',
                 'index_case'])

    hmaps_testing_rate = {'N_infected':{'student':np.nan, 'teacher':np.nan},
                 'R0':{'student':np.nan, 'teacher':np.nan}}

    for index_case in ['student', 'teacher']:
        cmap = np.zeros((len(testing_rates), len(school_types)))
        cmap_R0 = np.zeros((len(testing_rates), len(school_types)))
        for i, tpr in enumerate(testing_rates):
            for j, st in enumerate(school_types):
                bl_infected_agents = data.loc[st, 1.0, 1.0, index_case]\
                    ['infected_agents_mean']
                bl_R0 = data.loc[st, 1.0, 1.0, index_case]['R0_mean']
                cmap[i, j] = data.loc[st, tpr, tpr, index_case]\
                    ['infected_agents_mean'] / bl_infected_agents
                cmap_R0[i, j] = data .loc[st, tpr, tpr, index_case]['R0_mean']

        hmaps_testing_rate['N_infected'][index_case] = cmap
        hmaps_testing_rate['R0'][index_case] = cmap_R0
    
    return hmaps_testing_rate


def build_class_size_reduction_heatmaps(class_size_reduction_data, 
        class_size_reductions, school_types=school_types):
    
    data = class_size_reduction_data[\
            (class_size_reduction_data['student_screen_interval'] == 'never') &\
            (class_size_reduction_data['teacher_screen_interval'] == 'never') &\
            (class_size_reduction_data['student_mask'] == False) &\
            (class_size_reduction_data['teacher_mask'] == False) &\
            (class_size_reduction_data['ventilation_modification'] == 1.0)]

    data = data.set_index(\
                ['school_type', 'class_size_reduction', 'index_case'])


    hmap_class_size_reduction = {'N_infected':{'student':np.nan, 'teacher':np.nan},
                 'R0':{'student':np.nan, 'teacher':np.nan}}

    for index_case in ['student', 'teacher']:
        cmap = np.zeros((len(class_size_reductions), len(school_types)))
        cmap_R0 = np.zeros((len(class_size_reductions), len(school_types)))
        for i, csr in enumerate(class_size_reductions):
            for j, st in enumerate(school_types):
                bl_infected_agents = data.loc[st, 0.5, index_case]\
                    ['infected_agents_mean']
                bl_R0 = data.loc[st, 0.5, index_case]['R0_mean']
                cmap[i, j] = data.loc[st, csr, index_case]['infected_agents_mean'] /\
                    bl_infected_agents
                cmap_R0[i, j] = data.loc[st, csr, index_case]['R0_mean']

        hmap_class_size_reduction['N_infected'][index_case] = cmap
        hmap_class_size_reduction['R0'][index_case] = cmap_R0
        
    return hmap_class_size_reduction


def build_ventilation_efficiency_heatmaps(ventilation_efficiency_data, 
        ventilation_efficiencies, school_types=school_types):
    
    data = ventilation_efficiency_data[\
            (ventilation_efficiency_data['student_screen_interval'] == 'never') &\
            (ventilation_efficiency_data['teacher_screen_interval'] == 'never') &\
            (ventilation_efficiency_data['student_mask'] == False) &\
            (ventilation_efficiency_data['teacher_mask'] == False) &\
            (ventilation_efficiency_data['class_size_reduction'] == 0.0)]

    data = data.set_index(\
                ['school_type', 'ventilation_modification', 'index_case'])


    hmaps_ventilation_efficiency = {
        'N_infected':{'student':np.nan, 'teacher':np.nan},
        'R0':{'student':np.nan, 'teacher':np.nan}
    }

    for index_case in ['student', 'teacher']:
        cmap = np.zeros((len(ventilation_efficiencies), len(school_types)))
        cmap_R0 = np.zeros((len(ventilation_efficiencies), len(school_types)))
        for i, ve in enumerate(ventilation_efficiencies):
            for j, st in enumerate(school_types):
                bl_infected_agents = data.loc[st, 0.36, index_case]\
                    ['infected_agents_mean']
                bl_R0 = data.loc[st, 0.36, index_case]['R0_mean']
                cmap[i, j] = data.loc[st, ve, index_case]['infected_agents_mean'] / \
                    bl_infected_agents
                cmap_R0[i, j] = data .loc[st, ve, index_case]['R0_mean']

        hmaps_ventilation_efficiency['N_infected'][index_case] = cmap
        hmaps_ventilation_efficiency['R0'][index_case] = cmap_R0

    return hmaps_ventilation_efficiency


def build_mask_efficiency_heatmaps(mask_efficiency_data, 
        mask_efficiencies_exhale, mask_efficiencies_inhale, 
        school_types=school_types):

    data = mask_efficiency_data[\
        (mask_efficiency_data['student_screen_interval'] == 'never') &\
        (mask_efficiency_data['teacher_screen_interval'] == 'never') &\
        (mask_efficiency_data['student_mask'] == True) &\
        (mask_efficiency_data['teacher_mask'] == True) &\
        (mask_efficiency_data['class_size_reduction'] == 0.0) &\
        (mask_efficiency_data['ventilation_modification'] == 1.0)]

    data = data.set_index(\
                ['school_type', 'mask_efficiency_exhale', 'mask_efficiency_inhale',
                 'index_case'])

    hmaps_mask_efficiency = {'N_infected':{'student':np.nan, 'teacher':np.nan},
                 'R0':{'student':np.nan, 'teacher':np.nan}}

    for index_case in ['student', 'teacher']:
        cmap = np.zeros((len(mask_efficiencies_exhale), len(school_types)))
        cmap_R0 = np.zeros((len(mask_efficiencies_exhale), len(school_types)))
        for i, mee, mei in zip(range(len(mask_efficiencies_exhale)),
                            mask_efficiencies_exhale, mask_efficiencies_inhale):
            for j, st in enumerate(school_types):
                bl_infected_agents = data.loc[st, 0.7, 0.5, index_case]\
                    ['infected_agents_mean']
                bl_R0 = data.loc[st, 0.7, 0.5, index_case]['R0_mean']
                cmap[i, j] = data.loc[st, mee, mei, index_case]['infected_agents_mean'] / \
                    bl_infected_agents
                cmap_R0[i, j] = data .loc[st, mee, mei, index_case]['R0_mean']

        hmaps_mask_efficiency['N_infected'][index_case] = cmap
        hmaps_mask_efficiency['R0'][index_case] = cmap_R0

    return hmaps_mask_efficiency


def build_added_friendship_contacts_heatmaps(added_friendship_contacts_data, 
        friendship_ratios, school_types=school_types):
    
    data = added_friendship_contacts_data[\
            (added_friendship_contacts_data['student_screen_interval'] == 'never') &\
            (added_friendship_contacts_data['teacher_screen_interval'] == 'never') &\
            (added_friendship_contacts_data['student_mask'] == False) &\
            (added_friendship_contacts_data['teacher_mask'] == False) &\
            (added_friendship_contacts_data['class_size_reduction'] == 0.0) &\
            (added_friendship_contacts_data['ventilation_modification'] == 1.0)]

    data = data.set_index(\
                ['school_type', 'friendship_ratio', 'index_case'])

    hmaps_added_friendship_contacts = {
        'N_infected':{'student':np.nan, 'teacher':np.nan},
        'R0':{'student':np.nan, 'teacher':np.nan}
    }

    for index_case in ['student', 'teacher']:
        cmap = np.zeros((len(friendship_ratios), len(school_types)))
        cmap_R0 = np.zeros((len(friendship_ratios), len(school_types)))
        for i, fr in zip(range(len(friendship_ratios)), friendship_ratios):
            for j, st in enumerate(school_types):
                bl_infected_agents = data.loc[st, 0.0, index_case]\
                    ['infected_agents_mean']
                bl_R0 = data.loc[st, 0.0, index_case]['R0_mean']
                cmap[i, j] = data.loc[st, fr, index_case]['infected_agents_mean'] / \
                    bl_infected_agents
                cmap_R0[i, j] = data .loc[st, fr, index_case]['R0_mean']

        hmaps_added_friendship_contacts['N_infected'][index_case] = cmap
        hmaps_added_friendship_contacts['R0'][index_case] = cmap_R0
    
    return hmaps_added_friendship_contacts


def plot_heatmaps(axes, heatmaps, ylabel, yticklabels, indicator_ypos, colors,
                 X_min=0, X_max=10, R_min=0, R_max=9, title=False, xticks=False,
                 school_types=school_types):
    
    images = {'N_infected':{'student':0, 'teacher':0},
              'R0':{'student':0}, 'teacher':0}
    xticklabels = ['primary', '+ daycare', 'low. sec.', '+ daycare', 'up. sec.',
                   'secondary']
    
    for i, observable, cmap in zip(range(2), ['N_infected', 'R0'],
                        [get_continuous_cmap(colors), plt.get_cmap('coolwarm')]):

        for j, index_case in enumerate(['student', 'teacher']):
            ax = axes[2*i + j]
        
            if observable == 'N_infected':
                img = ax.imshow(heatmaps[observable][index_case],
                             vmin=X_min, vmax=X_max, cmap=cmap)
            else:
                img = ax.imshow(heatmaps[observable][index_case], 
                    clim = (R_min, R_max), 
                    norm = MidpointNormalize(midpoint=1, vmin=R_min, vmax=R_max),
                    cmap = cmap)
            images[observable][index_case] = img

            if 2*i + j == 0:
                ax.set_yticks(range(len(yticklabels)))
                ax.set_yticklabels(yticklabels)
                ax.set_ylabel(ylabel, fontsize=16)
            else:
                ax.set_yticks([])
                ax.set_yticklabels([])
                
            if xticks:
                ax.set_xticks(range(len(school_types)))
                ax.set_xticklabels(['primary', '+ daycare', 'low. sec.',
                                '+ daycare', 'up. sec.', 'secondary'], fontsize=12)
                ax.tick_params(axis='x', rotation=90)
            else:
                ax.set_xticks([])
            if title:
                ax.set_title('index case: {}'.format(index_case), fontsize=11)

    # draw a box around the conservative estimate
    for ax in axes:
        rect = Rectangle((-0.41, indicator_ypos), 5.85, 1,
                     linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    return images