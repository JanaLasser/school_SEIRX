#!/bin/bash

uptime
echo -n "start: "
date

N_runs=500
          
max_tasks=200                ## number of tasks per node.
running_tasks=0              ## initialization
src=../../data/contact_networks/reduced_class_size
dst=../../data/sensitivity_analysis/simulation_results/worst_case


for stype in primary primary_dc lower_secondary lower_secondary_dc upper_secondary secondary
	do
	
	for m_idx in $(seq 0 27)
		do
		running_tasks=`ps -C python --no-headers | wc -l`
		
		while [ "$running_tasks" -ge "$max_tasks" ]
			do
			sleep 5
			running_tasks=`ps -C python --no-headers | wc -l`
		done

		echo "*********************"
		echo run_data_creation_worst_case.py $stype $N_runs $m_idx $src $dst
		python3 run_data_creation_worst_case.py $stype $N_runs $m_idx $src $dst &
		echo "*********************"
		sleep 1
		
	done
done

wait

echo -n "end: "
date