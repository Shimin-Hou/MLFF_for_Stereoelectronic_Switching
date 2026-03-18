> ml potential (DeePMD-kit v3.0.0a0)
> 
The file "input.json" is an input file for one round of the active learning iteration. Please refer to the SI for details of the training process. This file is intended only to illustrate the model structure.

> molecular dynamics
> 
A total of 45 molecular dynamics simulation trajectory files are included. For parameter settings, please refer to "input.lammps".

> stretch-compress
> 
This is statistical data from the stretching-compression cycle simulations, including dihedral angles and corresponding conductance, as well as data processing scripts.

> transmission_calculate (Transiesta)
> 
The settings related to transmission calculations are provided here.

> transmission_fitting (torch 2.0.0.post200, python 3.10.13)
> 
This is an example of a processing script used for machine learning fitting of transmission spectrum. Run "train.py" to perform the fitting, and execute "plt_fit.py" to output the final results with the suffix "_negf_fitting".

[1] Andolina, C. M.;  Bon, M.;  Passerone, D.;Saidi, W. A. Robust, Multi-Length-Scale, Machine Learning Potential for Ag–Au Bimetallic Alloys from Clusters to Bulk Materials. J. Phys. Chem. C 2021, 125 (31), 17438-17447.
