# dustdriver

DUST2DUST.py is the main file, which calls callSALT2mu.py when necessary.

In order to run DUST2DUST.py, a configuration file is needed. IN_DUST2DUST.yaml is provided as an example input file. 


These two things are including for using in debugging - end user should not touch them!
`source RUNTEST_SUBPROCESS_BS20DATA` - runs SALT2mu once on the real data

`source RUNTEST_SUBPROCESS_SIM` - runs interactive SALT2mu job and shows you how the python code talks to C
