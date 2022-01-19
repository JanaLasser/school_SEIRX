import pandas as pd
from os.path import join
import os
import sys
import json
import data_creation_functions as dcf

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
    if test == "test":
        test = True
    else:
        print("unknown command line parameter {}".format(test))
except IndexError:
    test = False

## I/O
# source of the contact networks for the calibration runs. There is a randomly
# generated contact network for each run in the ensemble.
contact_network_src = "../../data/contact_networks/representative_schools"
# destination of the data for the overall statistics generated in the
# calibration run
dst = "../../data/vaccinations_omicron/simulation_results"


## simulation settings
with open("params/vaccinations_measures.json", "r") as fp:
    measures = json.load(fp)
with open("params/vaccinations_simulation_parameters.json", "r") as fp:
    simulation_params = json.load(fp)
with open("params/vaccinations_school_characteristics.json", "r") as fp:
    school_characteristics = json.load(fp)


## parameter grid for which simulations will be run
screening_params = pd.read_csv(join("screening_params", "vaccinations.csv"))

params = [
    (
        N_runs,
        st,
        row["index_case"],
        dcf.format_none_column(row["s_screen_interval"]),
        dcf.format_none_column(row["t_screen_interval"]),
        row["s_mask"],
        row["t_mask"],
        row["half_classes"],
        row["ventilation_modification"],
        row["student_vaccination_ratio"],
        row["teacher_vaccination_ratio"],
        row["family_member_vaccination_ratio"],
    )
    for st in school_types
    for i, row in screening_params.iterrows()
]

if test:
    params = params[0:10]
    print(
        "This is a testrun, scanning only {} parameters with {} runs each.".format(
            len(params), N_runs
        )
    )
else:
    print(
        "There are {} parameter combinations to sample with {} runs each.".format(
            len(params), N_runs
        )
    )


## simulation runs
def run(params):

    # extract the simulation parameters from the parameter list
    (
        N_runs,
        school_type,
        index_case,
        s_screen_interval,
        t_screen_interval,
        student_mask,
        teacher_mask,
        half_classes,
        ventilation_mod,
        student_vaccination_ratio,
        teacher_vaccination_ratio,
        family_member_vaccination_ratio,
    ) = params

    try:
        os.mkdir(join(dst, school_type))
    except FileExistsError:
        pass

    ttype = "same_day_antigen"

    # run the ensemble with the given parameter combination and school type
    ensmbl_results = run_ensemble(
        N_runs,
        school_type,
        measures,
        simulation_params,
        school_characteristics,
        contact_network_src,
        dst,
        index_case,
        ttype=ttype,
        s_screen_interval=s_screen_interval,
        t_screen_interval=t_screen_interval,
        student_mask=student_mask,
        teacher_mask=teacher_mask,
        half_classes=half_classes,
        ventilation_mod=ventilation_mod,
        student_vaccination_ratio=student_vaccination_ratio,
        teacher_vaccination_ratio=teacher_vaccination_ratio,
        family_member_vaccination_ratio=family_member_vaccination_ratio,
    )

    ensmbl_results["school_type"] = school_type
    ensmbl_results["index_case"] = index_case
    ensmbl_results["test_type"] = ttype
    ensmbl_results["student_screen_interval"] = s_screen_interval
    ensmbl_results["teacher_screen_interval"] = t_screen_interval
    ensmbl_results["student_mask"] = student_mask
    ensmbl_results["teacher_mask"] = teacher_mask
    ensmbl_results["half_classes"] = half_classes
    ensmbl_results["ventilation_mod"] = ventilation_mod
    ensmbl_results["student_vaccination_ratio"] = student_vaccination_ratio
    ensmbl_results["teacher_vaccination_ratio"] = teacher_vaccination_ratio
    ensmbl_results["family_member_vaccination_ratio"] = family_member_vaccination_ratio

    return ensmbl_results


# figure out which host we are running on and determine number of cores to
# use for the parallel programming
hostname = socket.gethostname()
if hostname == "desiato":
    number_of_cores = 200  # desiato
    print("running on {}, using {} cores".format(hostname, number_of_cores))
elif hostname == "T14s":
    number_of_cores = 14  # laptop
    print("running on {}, using {} cores".format(hostname, number_of_cores))
elif hostname == "marvin":
    number_of_cores = 28  # marvin
    print("running on {}, using {} cores".format(hostname, number_of_cores))
elif hostname == "medea.isds.tugraz.at":
    number_of_cores = 200  # marvin
    print("running on {}, using {} cores".format(hostname, number_of_cores))
else:
    print("unknown host")


# run the simulation in parallel on the available cores
pool = Pool(number_of_cores)
results = pd.DataFrame()
for ensmbl_results in tqdm(
    pool.imap_unordered(func=run, iterable=params), total=len(params)
):
    results = results.append(ensmbl_results, ignore_index=True)

# turn off your parallel workers
pool.close()

results = results.reset_index(drop=True)

index_cols = [
    "school_type",
    "index_case",
    "test_type",
    "student_screen_interval",
    "teacher_screen_interval",
    "student_mask",
    "teacher_mask",
    "half_classes",
    "ventilation_mod",
    "student_vaccination_ratio",
    "teacher_vaccination_ratio",
    "family_member_vaccination_ratio",
]
other_cols = [c for c in results.columns if c not in index_cols]
results = results[index_cols + other_cols]

results.to_csv(join(dst, "vaccinations_{}_{}.csv".format(N_runs, st)), index=False)
