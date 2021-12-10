# Assessing the impact of SARS-CoV-2prevention measures in Austrian schools by means of agent-based simulations calibrated to clustertracing data
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
4. [Analysis of vaccination scenarios](https://github.com/JanaLasser/school_SEIRX/tree/main/code/vaccinations)
5. [Visualizations](https://github.com/JanaLasser/school_SEIRX/tree/main/code/visualizations)

Each of the four parts has its own ```README``` file, giving instructions on how to reproduce the simulations and analyses presented in the paper.

**Note**: to run the scripts listed here, you will need to download the necessary data from the [OSF repository](https://osf.io/mde4k/) corresponding to this project. Please download the folder ```data``` from the repository and place it in the same directory of the repository as this README file.

## Requirements
This project requires Python and Jupyter Notebooks as well as the scseirx Python package (v1.4.1) to run. The scseirx python package can be installed using pip. This should satisfy all other package dependencies.

```pip install -Iv scseirx==1.4.1```

The installation of the package takes about one minute on a modern laptop with a fast internet connection, depending on how many dependencies need to be downloaded.

In the [README](https://github.com/JanaLasser/agent_based_COVID_SEIRX) of the scseirx simulation repository you can find
* A detailed description of the simulation design and implementation
* Easy-to-follow usage examples for the simulation package (specifically relevant for the school application is [this](https://github.com/JanaLasser/agent_based_COVID_SEIRX/blob/v1.4.1/src/scseirx/example_school.ipynb) example notebook).
* Installation instructions for linux


