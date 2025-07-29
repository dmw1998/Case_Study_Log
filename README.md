[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![CasADi](https://img.shields.io/badge/CasADi-3.7.0-blue.svg)](https://web.casadi.org/)
[![Chaospy](https://img.shields.io/badge/Chaospy-4.3.20-orange.svg)](https://chaospy.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/Numpy-2.3.1-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.3-yellow.svg)](https://matplotlib.org/)
[![Last Commit](https://img.shields.io/github/last-commit/dmw1998/Case_Study_Log)](https://github.com/dmw1998/Case_Study_Log/commits/main)

# Case Study

This project is part of the TUM Case Study Seminar S3, focusing on optimal control of aircraft trajectories under uncertain wind shear conditions. This code is written in Python 3.12 and uses the CasADi and Chaospy library.

This repository contains the **initial complete and executable model** independently developed by **Miaowen Dong** during a collaborative group project. It served as the **first working version** and provided the foundation for further development, including model extensions and interface migration carried out by other team members.

## Overview

At an early stage of the project, when upstream modeling components were incomplete or non-functional, this repository was created to ensure continued progress. The model includes a structured and parameterized system, which was later extended to support more complex scenarios (e.g., multiple plans model by Jakob Dilen [MrGoomb0](https://github.com/MrGoomb0) and uncertain wind with one or two stochastic variables by myself). Two failure probability estimation methods (Monte Carlo and subset simulation) were also developed and evaluated within this framework.

**Key features of this repository include:**
- A fully functional, standalone model built from scratch.
- Two smoothed wind profile for solver.
- The foundation for all subsequent extensions.
- Design and execution of comparative experiments and visualizations.
- A modular codebase that facilitates theoretical enhancements and system scaling.

## Key Components

- `All_in_Poster.ipynb` – Summary of modeling progress up to the poster stage.
- `numerical_experiments.ipynb` – Full experiment notebook.
  
  **Note**: The following parts were developed by Jakob Dilen ([MrGoomb0](https://github.com/MrGoomb0)):  
    – the multiple plans model,  
    – Method 2.1: embedded error estimation with a posteriori mesh halving, and  
    – Method 2.2: embedded error estimation using Runge-Kutta pairs.  
  The rest of the modeling, simulation, and result framework in this notebook was developed by Miaowen Dong.
  
- `mc_failure_probability.py` – Monte Carlo experiments with three solvers (later replaced).
- `sus_new.py` – Subset simulation for failure probability estimation.
- `visualization.py` – Visualization scripts for subset simulation results.

## Contribution Summary

- **Developed the first fully functional version of the model**, enabling downstream progress across the team.
- **Designed and implemented all core components**, including wind models, optimimal control solver framework, Monte Carlo and Subset Simulation estimation, and all numerical experiments except for those listed below.
- **The multiple plans model and both embedded error estimation methods (a posteriori mesh halving and Runge-Kutta pairs) were contributed by Jakob Dilen [MrGoomb0](https://github.com/MrGoomb0).**
- **Delivered a modular, well-documented code structure** that supported collaborative testing, debugging, and reuse.


## Contact

For questions about the model architecture, experiments, or contributions, feel free to contact [Miaowen Dong](mailto:miaowen.dong@tum.de) or open an issue on this repository.
