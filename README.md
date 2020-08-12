# dustdriver

`source RUNTEST_SUBPROCESS_BS20DATA` - runs SALT2mu once on the real data

`source RUNTEST_SUBPROCESS_SIM` - runs an interactive SALT2mu job and shows you how the python code talks to C

`runparallelemcee.py` - is the code that gets submitted to the batch system which runs the mcmc and computes the likelihood

`slurm.job` - job submission script

Currently I have an environment called "dillon" which has `corner.py` installed... can change to whatever...

