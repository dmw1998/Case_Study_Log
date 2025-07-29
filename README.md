# Case Study

This repository contains the **initial complete and executable model** independently developed by **Miaowen Dong** during a collaborative group project. It served as the **first working version** and provided the foundation for further development, including model extensions and interface migration carried out by other team members.

## 🔍 Overview

At an early stage of the project, when upstream modeling components were incomplete or non-functional, this repository was created to ensure continued progress. The model includes a structured and parameterized system, which was later extended to support more complex scenarios (e.g., multiple aircraft plans by [MrGoomb0](https://github.com/MrGoomb0) and uncertain wind with one or two stochastic variables by myself). Two failure probability estimation methods (Monte Carlo and subset simulation) were also developed and evaluated within this framework.

**Key features of this repository include:**
- A fully functional, standalone model built from scratch.
- Two smoothed wind profile for solver.
- The foundation for all subsequent extensions.
- Design and execution of comparative experiments and visualizations.
- A modular codebase that facilitates theoretical enhancements and system scaling.

## Contribution Summary

- **Developed the first fully functional version of the model**, enabling downstream progress across the team.
- **Provided the structural basis for extended systems**, including the multi-plane model implemented via the `Opti` interface (by [MrGoomb0](https://github.com/MrGoomb0)), which builds directly upon this initial version.
- **Designed and implemented key comparative experiments**, including scenario setup, evaluation scripts, and result visualizations.
- **Delivered a modular, well-documented code structure** that supported collaborative testing, debugging, and reuse.

## Contact

For questions about the model architecture, experiments, or contributions, feel free to contact [Miaowen Dong](mailto:miaowen.dong@tum.de) or open an issue on this repository.
