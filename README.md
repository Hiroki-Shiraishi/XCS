# XCS (Accuracy-based LCS)

Overview
---
This is based ENTIRELY on "An algorithmic description of XCS". You can solve N-bit Multiplexer problem (default: N = 6).

XCS only supports binary input. You can find support for real values input here: https://github.com/Hiroki-Shiraishi/XCSR

```
python main.py
```
If you want to change the hyperparameters, refer to "parameters.py", please.


Output Example
---
```
[ Settings ]
    Environment = 6-bit Multiplexer problem
  NumIterations = 10000
NumCondenseIter = 10000

[ XCS General Parameters ]
              N = 400
           beta = 0.2
          alpha = 0.1
      epsilon_0 = 10
             nu = 5
          gamma = 0.71
       theta_GA = 25
            chi = 0.8
             mu = 0.04
      theta_del = 20
          delta = 0.1
      theta_sub = 20
            P_# = 0.33
            p_I = 0.01
      epsilon_I = 0.01
            F_I = 0.01
        p_explr = 1.0
      theta_mna = 2
doGAsubsumption = True
doASSubsumption = True
crossoverMethod = two-point

[ XCS Optional Settings]
            tau = 0.4

  Iteration      Reward      SysErr     PopSize  CovOccRate
=========== =========== =========== =========== ===========
       1000     884.000     255.798     116.564  0.00037000
       2000     992.000      73.229      93.467  0.00000000
       3000    1000.000      17.541      48.790  0.00000000
       4000    1000.000      18.085      42.490  0.00000000
       5000    1000.000      15.038      42.337  0.00000000
       6000    1000.000      12.609      38.382  0.00000000
       7000    1000.000      14.552      44.313  0.00000000
       8000    1000.000      12.653      42.724  0.00000000
       9000    1000.000      13.614      44.052  0.00000000
      10000    1000.000      18.459      47.101  0.00000000
      11000    1000.000       1.035      31.727  0.00000000
      12000    1000.000       0.000      29.000  0.00000000
      13000    1000.000       0.000      29.000  0.00000000
      14000    1000.000       0.000      28.375  0.00000000
      15000    1000.000       0.000      28.000  0.00000000
      16000    1000.000       0.000      28.000  0.00000000
      17000    1000.000       0.000      28.000  0.00000000
      18000    1000.000       0.000      28.000  0.00000000
      19000    1000.000       0.000      28.000  0.00000000
      20000    1000.000       0.000      28.000  0.00000000
```