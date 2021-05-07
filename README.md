# Assessing the impact of SARS-CoV-2prevention measures in schools by means ofagent-based simulations calibrated to clustertracing data
**Author: Jana Lasser, Graz University of Technology (jana.lasser@tugraz.at)**

Simulations using the [scseirx](https://pypi.org/project/scseirx/) (small-community-SEIRX) package to simulate disease spread in small communities and under a range of different interventions. This repository is an application to schools, presented in [this](https://doi.org/10.1101/2021.04.13.21255320) preprint.

*This software is under development and intended to respond rapidly to the current situation. Please use it with caution and bear in mind that there might be bugs.*


Reference:  

_Lasser, J., Sorger, J., Richter, L., Thurner, S., Schmid, D., Klimek, P. (2021). Assessing the impact of SARS-CoV-2prevention measures in schools by means ofagent-based simulations calibrated to clustertracing data. [DOI](https://doi.org/10.1101/2021.04.13.21255320): 10.1101/2021.04.13.21255320_

## Contents
The content of this repository is structured into four main parts, also reflected in the folder structure of the ```code``` subfolder.
1. [Model calibration](https://github.com/JanaLasser/school_SEIRX/tree/main/code/calibration)
2. [Analysis of intervention measures](https://github.com/JanaLasser/school_SEIRX/tree/main/code/intervention_measures)
3. [Sensitivity analysis](https://github.com/JanaLasser/school_SEIRX/tree/main/code/sensitivity_analysis)
4. [Visualizations](https://github.com/JanaLasser/school_SEIRX/tree/main/code/visualizations)

Each of the four parts has its own ```README``` file, giving instructions on how to reproduce the simulations and analyses presented in the paper.

## Requirements
This project requires Python and Jupyter Notebooks to run. The scseirx python package can be installed using pip. This should satisfy all other package dependencies.

```pip install scseirx```

In the [README](https://github.com/JanaLasser/agent_based_COVID_SEIRX) of the scseirx-simulation repository you can find
* A detailed description of the simulation design and implementation
* Easy-to-follow usage examples of the simulation package
* Installation instructions for linux


