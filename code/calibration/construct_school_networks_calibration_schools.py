import networkx as nx
import pandas as pd
from os.path import join
import sys
import socket

# network construction utilities
from scseirx import construct_school_network as csn

# parallelisation functionality
from multiprocess import Pool
import psutil
from tqdm import tqdm

school_types = [sys.argv[1]]
N_networks = sys.argv[2]
dst = '../../data/contact_networks/calibration'


school_params = [(st, i, N_floors) for st in school_types \
                                      for i in range(N_networks)]


# different age structures in Austrian school types
age_brackets = {'primary':[6, 7, 8, 9],
                'primary_dc':[6, 7, 8, 9],
                'lower_secondary':[10, 11, 12, 13],
                'lower_secondary_dc':[10, 11, 12, 13],
                'upper_secondary':[14, 15, 16, 17],
                'secondary':[10, 11, 12, 13, 14, 15, 16, 17],
                'secondary_dc':[10, 11, 12, 13, 14, 15, 16, 17]
               }

# average number of classes per school type and students per class
school_characteristics = {
    # Primary schools
    # Volksschule: schools 3033, classes: 18245, students: 339382
    'primary':            {'classes':8, 'students':19},
    'primary_dc':         {'classes':8, 'students':19},
    
    # Lower secondary schools
    # Hauptschule: schools 47, classes 104, students: 1993
    # Mittelschule: schools 1131, classes: 10354, students: 205905
    # Sonderschule: schools 292, classes: 1626, students: 14815
    # Total: schools: 1470, classes: 12084, students: 222713
    'lower_secondary':    {'classes':8, 'students':18},
    'lower_secondary_dc': {'classes':8, 'students':18},
    
    # Upper secondary schools
    # Oberstufenrealgymnasium: schools 114, classes 1183, students: 26211
    # BMHS: schools 734, classes 8042, students 187592
    # Total: schools: 848, classes 9225, students: 213803
    'upper_secondary':    {'classes':10, 'students':23}, # rounded down from 10.8 classes
    
    # Secondary schools
    # AHS Langform: schools 281, classes 7610, students 179633
    'secondary':          {'classes':28, 'students':24}, # rounded up from 27.1 classes
    'secondary_dc':       {'classes':28, 'students':24} # rounded up from 27.1 classes
}

# given the precondition that the family has at least one child, how many
# children does the family have?
p_children = {1:0.4815, 2:0.3812, 3:0.1069, 4:0.0304}

# probability of being a single parent, depending on the number of children
p_parents = {1:{1:0.1805, 2:0.8195},
             2:{1:0.1030, 2:0.8970},
             3:{1:0.1174, 2:0.8826},
             4:{1:0.1256, 2:0.8744}
            }

# probability of a household having a certain size, independent of having a child
teacher_p_adults = {1:0.4655, 2:0.5186, 3:0.0159}
teacher_p_children = {1:{0:0.8495, 1:0.0953, 2:0.0408, 3:0.0144},
                      2:{0:0.4874, 1:0.2133, 2:0.2158, 3:0.0835},
                      3:{0:1, 1:0, 2:0, 3:0}}

contact_map = {
    'student_household':'close', 
    'student_student_intra_class':'far',
    'student_student_table_neighbour':'intermediate',
    'student_student_daycare':'far',
    'teacher_household':'close',
    'teacher_teacher_short':'far', 
    'teacher_teacher_long':'intermediate',
    'teacher_teacher_team_teaching':'intermediate',
    'teacher_teacher_daycare_supervision':'intermediate',
    'teaching_teacher_student':'far',
    'daycare_supervision_teacher_student':'far'
}
# Note: student_student_daycare overwrites student_student_intra_class and
# student_student_table_neighbour

# Note: teacher_teacher_daycare_supervision and teacher_teacher_team_teaching 
# overwrite teacher_teacher_short and teacher_teacher_long

r_teacher_friend = 0.059
r_teacher_conversation = 0.255

def run(params):
    school_type, i, N_floors = params
    
    N_classes = school_characteristics[school_type]['classes']
    class_size = school_characteristics[school_type]['students']
    
    school_name = '{}_classes-{}_students-{}'.format(school_type,\
            N_classes, class_size)
    
    G, teacher_schedule, student_schedule = csn.compose_school_graph(\
                school_type, N_classes, class_size, N_floors, p_children,
                p_parents, teacher_p_adults, teacher_p_children, 
                r_teacher_conversation, r_teacher_friend)

    # map the link types to contact types
    csn.map_contacts(G, contact_map)

    # we do not need family members that are not siblings for calibration
    # purposes -> remove them to have less agents in the simulation and
    # speed up the calibration runs
    family_members = [n for n, tp in G.nodes(data='type') \
                if tp in ['family_member_student', 'family_member_teacher']]
    G.remove_nodes_from(family_members)

    # save the graph
    nx.readwrite.gpickle.write_gpickle(G, join(dst,'{}/{}_{}.bz2'\
                        .format(school_type, school_name, i)), protocol=4)

    # extract & save the node list
    node_list = csn.get_node_list(G)
    node_list.to_csv(join(dst,'{}/{}_node_list_{}.csv')\
                        .format(school_type, school_name, i), index=False)
    # save the schedule
    if i==1:
        for s, atype in zip([teacher_schedule, student_schedule],\
                            ['teachers', 'students']):
            s.to_csv(join(dst, '{}/{}_schedule_{}.csv'\
                        .format(school_type, school_name, atype)))
            
# in principle there is functionality in place to generate contacts
# between students in different classes, depending on the floor the
# classes are on. We currently don't use this functionality, as 
# schools all implement measures to keep between-class-contacts to
# a minimum- Therefore floor specifications are not important for our
# school layout and we just assume that all classes are on the same
# floor.
N_floors = 1

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
    
pool = Pool(number_of_cores)

for row in tqdm(pool.imap_unordered(func=run, iterable=school_params),
                total=len(school_params)):
    pass

# turn off your parallel workers 
pool.close()