# Fix-and-Propagate Heuristics Using Low-Precision First-Order LP Solutions for Large-Scale Mixed-Integer Linear Optimization

This repository contains the code used in [[1]](#1) and [[2]](#2) for solving large scale energy system models.

Experimental logs for [[1]](#1) are available in [[4]](#4) and the models used in [[1]](#1) and [[2]](#2) can be found in [[5]](#5) (and a set of Gurobi presolved models can be found in [[6]](#6)). The code is derived from [[3]](#3) with permission of the author.

## Compile

This code reuqires at least one of the solvers CPLEX (>= 12.10), COPT (>= 7.0), Gurobi (>= 11.0), or XPRESS; cmake >= 3.12 is required. As this code uses some C++20, the compiler will need to support this as well.

The solvers <SOLVER> (one of COPT, CPLEX, GUROBI, XPRESS) can be either linked in, e.g., 'fp_fom/solvers/<SOLVER>' with subfolders 'lib' and 'include' or the respective solver directory can be specified via `cmake .. -D<SOLVER>DIR (see also 'extern/utils/cmake' for the cmake lookup files). The build workflow is

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -CPLEXDIR=${PATH_TO_CPLEX_INSTALLATION} -DXPRESSDIR=${PATH_TO_XPRESS_INSTALLATION} -DGUROBIDIR=${PATH_TO_GUROBI_INSTALLATION} -DXPRESSDIR=${PATH_TO_XPRESS_INSTALLTION}
make -j10
```

This will generate an executable called 'fp_main'.

## Usage

To query usage of the binary, run either 'fp_main' or 'fp_main -h'. This will display instructions on how to run fp_main and how to set options:
```
Usage: fp_main instance_file [options]

Available options (case sensitive!) are:
Parameters and their defaults:

seed=20250101
timeLimit=1200
threads=32
maxTries=1
enableOutput=true
displayInterval=500
propagate=true
repair=false
backtrackOnInfeas=true
maxConsecutiveInfeas=0.2
minNodes=100000
maxNodes=-1
maxLpSolved=1
maxSolutions=1
useOldBranching=false
preset=UNKNOWN
ranker=TYPE
valueChooser=RANDOM_LP
lpMethod=BARRIER
lpMethodFinal=BARRIER
mipPresolve=true
postsolve=false
writeSol=false
zeroObj=false
randomWalkProbability=0.75
maxRepairNonImprove=10
maxRepairSteps=100
solver=COPT
presolver=GUROBI
lpTol=1e-06
runPortfolio=false

Available options:

Available rankers are:
redcostsbreakfrac
lr
cliques
obj
infer_obj
dualsbreakfrac
type
typecl
random
locks
fracbreakduals
redcosts
cliquess
duals
fracbreakredcosts
frac

Available value choosers are:
split
round_int
random_up_down
random_lp
random
up
loose
infer_obj
down
bad_obj
good_obj

Available lp methods are:
fom
barrier_crossover
barrier
dual
primal

Available solvers are:
cplex
gurobi
xpress
copt

Presolve is available with GUROBI and CPLEX
Postsolve is available only with CPLEX
```

Generally, fp_main is executed as
```
fp_main instance_file [options]
```

For example, running PDLP based fix-and-propagate as the redcost strategy for ranking the variables during the fix-and-propagate run and using the lp solution value for picking a value to fix the variables to, run
```
fp_main <path_to_instance>/<instance_name> presolver=GUROBI solver=COPT lpMethod=FOM ranker=REDCOSTS valueChooser=RANDOM_LP
```

The best solution found will be written to a file called <instance_name>.sol only if either presolve is turned off or CPLEX is specified as PRESOLVER. Gurobi does not support postsolve. Further examples on how to utilize `fp_main` together with scripts for easy execution can be found in [[4]](#4). The inferred objective fix-and-propagate variant from [[2]](#2) can be run by setting `ranker=INFER_OBJ valueChooser=INFER_OBJ`.

A typical example looks like

```
[config]
gitHash = 041bc0d-dirty
probFile = /scratch/htc/nkempke/Instances/miso_2M/miso3.lp.gz
strategy = REDCOSTS_RANDOM_LP
solver = COPT
seed = 20250101
timeLimit = 7000
threads = 32
maxTries = 1
enableOutput = true
displayInterval = 500
propagate = true
repair = false
backtrackOnInfeas = true
maxConsecutiveInfeas = 0.2
minNodes = 100000
maxNodes = -1
maxLpSolved = 1
maxSolutions = 1
useOldBranching = false
preset=UNKNOWN
ranker = REDCOSTS
valueChooser = RANDOM_LP
lpMethod = FIRST_ORDER_METHOD
lpMethodFinal = BARRIER
mipPresolve = true
postsolve = false
randomWalkProbability = 0.75
maxRepairNonImprove = 10
maxRepairSteps = 100
zeroObj = false
solver = COPT
presolver = GUROBI
lpTol = 0.0001
runPortfolio = false

Creating GUROBI presolver.

Set parameter TokenServer to value "solice01.zib.de"
Reading the problem.
Read LP format model from file /scratch/htc/nkempke/Instances/miso_2M//miso3.lp.gz
Reading time = 0.78 seconds
obj: 928790 rows, 880606 columns, 2829976 nonzeros
Reading time = 0.783358102

Presolving the problem.
Original Problem:  #rows=928790 #cols=880606 #nnz=2829976
Gurobi Optimizer version 11.0.3 build v11.0.3rc0 (armlinux64 - "Ubuntu 22.04.5 LTS")

CPU model: ARM64
Thread count: 72 physical cores, 72 logical processors, using up to 32 threads

Presolve removed 665292 rows and 600801 columns
Presolve time: 1.80s
Set parameter TokenServer to value "solice01.zib.de"
Presolved Problem: #rows=263498 #cols=279805 #nnz=865025
Presolve time = 1.826911924
Converting the problem to COPT
Row classification:
CLIQUE: 0
CLIQUE_EQ: 0
CLIQUE_EQ_NEGATED: 0
SETCOVER: 0
CARD: 0
CARD_EQ: 0
DOUBLE_AGGR: 0
VBOUND: 10596
KNAPSACK: 0
KNAPSACK_EQ: 0
GENERIC: 252902

Col classification:
BINARIES: 2
INTEGERS: 35048 [-0,180] |131|
CONTINUOUS: 244755
SINGLETONS: 69859
OBJSUPPORT: 131195
UPROUNDABLE: 16
DNROUNDABLE: 0

Cliquetable: 0 cliques and 0 nonzeros
Impltable: 10596 implications

Solving initial LP relaxation
Using COPT FIRST_ORDER_METHOD to solve the LP relaxation.
Setting parameter 'Logging' to 1
Setting parameter 'TimeLimit' to 6997.04
Setting parameter 'Threads' to 32
Setting parameter 'Crossover' to 0
Setting parameter 'PDLPTol' to 0.0001
Setting parameter 'LpMethod' to 6
Model fingerprint: ebe87601

Using Cardinal Optimizer v7.2.3 on Linux (aarch64)
Hardware has 72 cores and 72 threads. Using instruction set ARMV8 (30)
Minimizing an LP problem

The original problem has:
    263498 rows, 279805 columns and 865025 non-zero elements
The presolved problem has:
    254738 rows, 271045 columns and 865025 non-zero elements

Hardware has 1 supported GPU device with CUDA 12.2
  GPU 0: NVIDIA GH200 480GB (CUDA capability 9.0)

Starting PDLP solver on GPU 0

Iterations       Primal.Obj         Dual.Obj        Gap  Primal.Inf  Dual.Inf    Time
         0  +4.61144375e+04  +4.61144375e+04  +0.00e+00    6.09e+03  0.00e+00   0.68s
      4000  +6.47795535e+04  +6.47840261e+04  +4.47e+00    7.53e+00  7.24e-07   1.27s
      6840  +6.47858562e+04  +6.47866417e+04  +7.86e-01    6.02e-01  1.27e-08   1.68s

PDLP status:                     OPTIMAL
PDLP iterations:                 6840
Primal objective:                6.47858562e+04
Dual objective:                  6.47866417e+04
Primal infeasibility (abs/rel):  6.02e-01 / 9.79e-05
Dual infeasibility (abs/rel):    1.27e-08 / 9.70e-12
Duality gap (abs/rel):           7.86e-01 / 6.06e-06

Postsolving

Solving finished
Status: Optimal  Objective: 6.4785856199e+04  Iterations: 0  Time: 1.76s
Setting parameter 'Crossover' to 1
Setting parameter 'RelGap' to 0.0001
Setting parameter 'BarPrimalTol' to 1e-08
Setting parameter 'BarDualTol' to 1e-08
Setting parameter 'LpMethod' to -1
Setting parameter 'PDLPTol' to 1e-06
LP time = 1.766841487
Running single heuristic!
DFS REDCOSTS-RANDOM_LP seed=20250101
REDCOSTS_RANDOM_LP: 500 nodes processed: depth=499 violation=0 elapsed=4.787820529
REDCOSTS_RANDOM_LP: 1000 nodes processed: depth=999 violation=0 elapsed=4.789004606
REDCOSTS_RANDOM_LP: 1500 nodes processed: depth=1499 violation=0 elapsed=4.789946536
REDCOSTS_RANDOM_LP: 2000 nodes processed: depth=1999 violation=0 elapsed=4.790863762
REDCOSTS_RANDOM_LP: 2500 nodes processed: depth=2499 violation=0 elapsed=4.7917895
REDCOSTS_RANDOM_LP: 3000 nodes processed: depth=2999 violation=0 elapsed=4.792699973
REDCOSTS_RANDOM_LP: 3500 nodes processed: depth=3499 violation=0 elapsed=4.793566927
REDCOSTS_RANDOM_LP: 4000 nodes processed: depth=3999 violation=0 elapsed=4.794384119
REDCOSTS_RANDOM_LP: 4500 nodes processed: depth=4499 violation=0 elapsed=4.795089471
REDCOSTS_RANDOM_LP: 5000 nodes processed: depth=4999 violation=0 elapsed=4.795855623
REDCOSTS_RANDOM_LP: 5500 nodes processed: depth=5499 violation=0 elapsed=4.79665736
REDCOSTS_RANDOM_LP: 6000 nodes processed: depth=5999 violation=0 elapsed=4.797404056
REDCOSTS_RANDOM_LP: 6500 nodes processed: depth=6499 violation=0 elapsed=4.798109279
REDCOSTS_RANDOM_LP: 7000 nodes processed: depth=6999 violation=0 elapsed=4.798825831
REDCOSTS_RANDOM_LP: 7500 nodes processed: depth=7499 violation=0 elapsed=4.799634704
REDCOSTS_RANDOM_LP: 8000 nodes processed: depth=7999 violation=0 elapsed=4.800273879
REDCOSTS_RANDOM_LP: 8500 nodes processed: depth=8499 violation=0 elapsed=4.800838621
REDCOSTS_RANDOM_LP: 9000 nodes processed: depth=8999 violation=0 elapsed=4.801391459
REDCOSTS_RANDOM_LP: 9500 nodes processed: depth=9499 violation=0 elapsed=4.801934696
REDCOSTS_RANDOM_LP: 10000 nodes processed: depth=9999 violation=0 elapsed=4.802498862
REDCOSTS_RANDOM_LP: 10500 nodes processed: depth=10499 violation=0 elapsed=4.803211478
REDCOSTS_RANDOM_LP: 11000 nodes processed: depth=10999 violation=0 elapsed=4.803968126
REDCOSTS_RANDOM_LP: 11500 nodes processed: depth=11499 violation=0 elapsed=4.804676422
REDCOSTS_RANDOM_LP: 12000 nodes processed: depth=11999 violation=0 elapsed=4.805334541
REDCOSTS_RANDOM_LP: 12500 nodes processed: depth=12499 violation=0 elapsed=4.80595938
REDCOSTS_RANDOM_LP: 13000 nodes processed: depth=12999 violation=0 elapsed=4.806592314
REDCOSTS_RANDOM_LP: 13500 nodes processed: depth=13499 violation=0 elapsed=4.807206785
REDCOSTS_RANDOM_LP: 14000 nodes processed: depth=13999 violation=0 elapsed=4.807806503
REDCOSTS_RANDOM_LP: 14500 nodes processed: depth=14499 violation=0 elapsed=4.808388462
REDCOSTS_RANDOM_LP: 15000 nodes processed: depth=14999 violation=0 elapsed=4.808957908
REDCOSTS_RANDOM_LP: 15500 nodes processed: depth=15499 violation=0 elapsed=4.809530266
REDCOSTS_RANDOM_LP: 16000 nodes processed: depth=15999 violation=0 elapsed=4.810102112
REDCOSTS_RANDOM_LP: 16500 nodes processed: depth=16499 violation=0 elapsed=4.810731527
REDCOSTS_RANDOM_LP: 17000 nodes processed: depth=16999 violation=0 elapsed=4.811358414
REDCOSTS_RANDOM_LP: 17500 nodes processed: depth=17499 violation=0 elapsed=4.812735708
REDCOSTS_RANDOM_LP: 18000 nodes processed: depth=17999 violation=0 elapsed=4.814329709
REDCOSTS_RANDOM_LP: 18500 nodes processed: depth=18499 violation=0 elapsed=4.816098688
REDCOSTS_RANDOM_LP: 19000 nodes processed: depth=18999 violation=0 elapsed=4.817777138
REDCOSTS_RANDOM_LP: 19500 nodes processed: depth=19499 violation=0 elapsed=4.818888926
REDCOSTS_RANDOM_LP: 20000 nodes processed: depth=19999 violation=0 elapsed=4.820086155
REDCOSTS_RANDOM_LP: 20500 nodes processed: depth=20499 violation=0 elapsed=4.821441754
REDCOSTS_RANDOM_LP: 21000 nodes processed: depth=20999 violation=0 elapsed=4.822601414
REDCOSTS_RANDOM_LP: 21500 nodes processed: depth=21499 violation=0 elapsed=4.823653041
REDCOSTS_RANDOM_LP: 22000 nodes processed: depth=21999 violation=0 elapsed=4.824643644
REDCOSTS_RANDOM_LP: 22500 nodes processed: depth=22499 violation=0 elapsed=4.825772872
REDCOSTS_RANDOM_LP: 23000 nodes processed: depth=22999 violation=0 elapsed=4.826946677
REDCOSTS_RANDOM_LP: 23461 nodes processed: depth=23459 violation=0 elapsed=4.82913502
REDCOSTS_RANDOM_LP: Time starting LP solve = 4.830046678
Setting parameter 'Logging' to 1
Setting parameter 'TimeLimit' to 6995.17
Setting parameter 'Crossover' to 0
Setting parameter 'RelGap' to 1e-06
Setting parameter 'BarPrimalTol' to 1e-06
Setting parameter 'BarDualTol' to 1e-06
Setting parameter 'LpMethod' to 2
Model fingerprint: d366436c

Using Cardinal Optimizer v7.2.3 on Linux (aarch64)
Hardware has 72 cores and 72 threads. Using instruction set ARMV8 (30)
Minimizing an LP problem

The original problem has:
    263498 rows, 279805 columns and 865025 non-zero elements
The presolved problem has:
    83438 rows, 160915 columns and 336545 non-zero elements

Starting barrier solver using 32 threads

Problem info:
Dualized in presolve:            No
Range of matrix coefficients:    [1e-03,8e+00]
Range of rhs coefficients:       [9e+00,2e+02]
Range of bound coefficients:     [6e-08,2e+05]
Range of cost coefficients:      [4e-04,1e+03]

Factor info:
Number of free columns:          0
Number of dense columns:         6
Number of matrix entries:        2.503e+05
Number of factor entries:        4.227e+05
Number of factor flops:          2.845e+06

Iter       Primal.Obj         Dual.Obj      Compl  Primal.Inf  Dual.Inf    Time
   0  +1.94294468e+08  -7.29130404e+09   1.05e+10    2.88e+04  6.60e+00   0.55s
   1  +1.04359577e+08  -1.30641237e+09   2.87e+09    1.53e+04  1.31e+00   0.57s
   2  +4.34362272e+07  -4.03988425e+08   1.06e+09    6.52e+03  4.20e-01   0.58s
   3  +4.88323414e+06  -1.15918498e+08   1.81e+08    6.60e+02  1.09e-01   0.60s
   4  +1.46379742e+06  -4.30306049e+07   5.26e+07    9.70e+01  3.82e-02   0.61s
   5  +4.93443432e+05  -9.69744485e+06   1.04e+07    5.59e-01  8.21e-03   0.63s
   6  +4.54329485e+05  -6.75092177e+06   7.32e+06    4.56e-01  5.79e-03   0.64s
   7  +4.38810095e+05  -3.96051566e+06   4.49e+06    3.75e-01  3.55e-03   0.66s
   8  +2.58244180e+05  -1.20195404e+06   1.48e+06    1.46e-01  1.16e-03   0.67s
   9  +1.61808248e+05  -4.04242897e+05   5.72e+05    6.81e-02  4.36e-04   0.69s
  10  +1.00082959e+05  -1.12180298e+05   2.14e+05    2.35e-02  1.66e-04   0.71s
  11  +8.94666986e+04  -2.16431659e+04   1.12e+05    1.62e-02  8.14e-05   0.72s
  12  +7.80164607e+04  +3.03996527e+04   4.78e+04    8.10e-03  3.33e-05   0.74s
  13  +7.55403692e+04  +3.79979372e+04   3.77e+04    5.63e-03  2.69e-05   0.75s
  14  +6.89903466e+04  +5.25420034e+04   1.65e+04    2.05e-03  1.25e-05   0.77s
  15  +6.68277044e+04  +5.95269570e+04   7.31e+03    9.48e-04  5.49e-06   0.78s
  16  +6.58790171e+04  +6.14321447e+04   4.45e+03    4.86e-04  3.54e-06   0.80s
  17  +6.57400288e+04  +6.20615538e+04   3.68e+03    4.15e-04  2.90e-06   0.82s
  18  +6.53581500e+04  +6.26659014e+04   2.69e+03    2.38e-04  2.27e-06   0.83s
  19  +6.52159252e+04  +6.30182046e+04   2.20e+03    1.65e-04  1.91e-06   0.85s
  20  +6.50486277e+04  +6.35963703e+04   1.45e+03    9.00e-05  1.31e-06   0.86s
  21  +6.49644973e+04  +6.38827864e+04   1.08e+03    5.29e-05  1.01e-06   0.88s
  22  +6.49265688e+04  +6.40151958e+04   9.12e+02    3.65e-05  8.68e-07   0.89s
  23  +6.48825983e+04  +6.42117372e+04   6.71e+02    1.89e-05  6.60e-07   0.91s
  24  +6.48696094e+04  +6.42894463e+04   5.80e+02    1.34e-05  5.78e-07   0.93s
  25  +6.48536790e+04  +6.43967752e+04   4.57e+02    8.32e-06  4.63e-07   0.94s
  26  +6.48406561e+04  +6.44851444e+04   3.56e+02    4.44e-06  3.68e-07   0.96s
  27  +6.48326088e+04  +6.45513389e+04   2.81e+02    2.26e-06  2.96e-07   0.97s
  28  +6.48265227e+04  +6.45890758e+04   2.37e+02    8.29e-07  2.55e-07   0.99s
  29  +6.48270566e+04  +6.46128646e+04   2.14e+02    5.82e-07  2.29e-07   1.00s
  30  +6.48229098e+04  +6.46846482e+04   1.38e+02    2.74e-07  1.50e-07   1.02s
  31  +6.48232150e+04  +6.47064418e+04   1.17e+02    1.53e-07  1.26e-07   1.04s
  32  +6.48195787e+04  +6.47371918e+04   8.24e+01    5.25e-08  9.13e-08   1.05s
  33  +6.48183615e+04  +6.47532444e+04   6.51e+01    9.31e-09  7.32e-08   1.07s
  34  +6.48176178e+04  +6.48052559e+04   1.24e+01    3.99e-09  1.35e-08   1.08s
  35  +6.48172238e+04  +6.48153989e+04   1.83e+00    1.01e-09  1.93e-09   1.10s
  36  +6.48168368e+04  +6.48164067e+04   4.30e-01    9.16e-10  4.83e-10   1.12s
  37  +6.48167768e+04  +6.48167272e+04   4.96e-02    1.49e-09  1.66e-11   1.13s
  38  +6.48167380e+04  +6.48167340e+04   3.97e-03    3.13e-08  5.78e-12   1.15s
  39  +6.48167375e+04  +6.48167374e+04   7.65e-05    6.19e-09  2.05e-12   1.16s

Barrier status:                  OPTIMAL
Primal objective:                6.48167375e+04
Dual objective:                  6.48167374e+04
Duality gap (abs/rel):           7.65e-05 / 1.18e-09
Primal infeasibility (abs/rel):  6.19e-09 / 3.85e-11
Dual infeasibility (abs/rel):    2.05e-12 / 2.63e-15
Postsolving

Solving finished
Status: Optimal  Objective: 6.4816737518e+04  Iterations: 0  Time: 1.18s
Setting parameter 'Crossover' to 1
Setting parameter 'RelGap' to 0.0001
Setting parameter 'BarPrimalTol' to 1e-08
Setting parameter 'BarDualTol' to 1e-08
Setting parameter 'LpMethod' to -1
Setting parameter 'PDLPTol' to 1e-06
REDCOSTS_RANDOM_LP: Time finished LP solve = 6.016145838
REDCOSTS_RANDOM_LP: LP time = 1.186086744
REDCOSTS_RANDOM_LP: LP solved to optimality!
REDCOSTS_RANDOM_LP: Objective 64816.73751771546
REDCOSTS_RANDOM_LP: Limits reached
Walked 0 times
tries = 1
Printing the solpool
Solution pool: 1 solutions
       n      Objective   RelViolation   AbsViolation   Feas     L1 dist    Time  FoundBy
       0       64816.74         0.0000         0.0000   true        0.00    6.03  REDCOSTS_RANDOM_LP
[results]
found = 1
primalBound = 64816.737507845086
minAbsViol = 0
time = 6.047253819
```

## References

<a id="1">[1]</a>
Nils-Christian Kempke and Thorsten Koch (2025).
Fix-and-Propagate Heuristics Using Low-Precision First-Order LP Solutions for Large-Scale Mixed-Integer Linear Optimization.
arXiv.
https://doi.org/10.48550/arXiv.2503.10344

<a id="2">[2]</a>
Kempke et al. (2025).
Developing heuristic solution techniques for large-scale unit commitment models.
arXiv.
https://doi.org/10.48550/arXiv.2502.19012

<a id="3">[3]</a>
Domenico Salvagnin, Roberto Roberti, and Matteo Fischetti. (2024).
A fix-propagate-repair heuristic for mixed integer programming.
In *Mathematical Programming Computation* (pp. 111-139). Springer Science and Business Media LLC.
https://doi.org/10.1007/s12532-024-00269-5

<a id="4">[4]</a>
Nils-Christian Kempke and Thorsten Koch (2025).
Experimental Logs and Results for ``Fix-and-Propagate Heuristics Using Low-Precision First-Order LP Solutions for Large-Scale Mixed-Integer Linear Optimization'''.
Zenodo.
https://doi.org/10.5281/zenodo.17831842

<a id="5">[5]</a>
Shima Sasanpour and Thomas Breuer (2025).
Mixed-Integer Energy System Optimization Models for Germany: Seven Spatial Aggregations with Monte Carlo–Sampled UC, Dispatch, and Expansion Scenarios.
Zenodo.
https://doi.org/10.5281/zenodo.17702892

<a id="6">[6]</a>
Nils-Christian Kempke and Thorsten Koch (2026).
Mixed-Integer Energy System Optimization Models for Germany: Seven Spatial Aggregations with Monte Carlo–Sampled UC, Dispatch, and Expansion Scenarios - Supplement: Gurobi Presolved Models.
Zenodo.
https://doi.org/10.5281/zenodo.18777743
