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
  Iteration      Reward      SysErr     PopSize  CovOccRate
=========== =========== =========== =========== ===========
       1000     878.000     262.436      98.310  0.36465000
       2000     973.000      99.140      90.186  0.00000000
       3000    1000.000      24.171      65.177  0.00000000
       4000    1000.000      14.513      44.458  0.00000000
       5000    1000.000      18.304      51.699  0.00000000
       6000    1000.000      15.602      44.802  0.00000000
       7000    1000.000      13.982      43.563  0.00000000
       8000    1000.000      14.334      45.391  0.00000000
       9000    1000.000      18.070      46.787  0.00000000
      10000    1000.000      16.197      46.373  0.00000000
      11000    1000.000       0.501      32.192  0.00000000
      12000    1000.000       0.000      30.000  0.00000000
      13000    1000.000       0.000      30.000  0.00000000
      14000    1000.000       0.000      29.841  0.00000000
      15000    1000.000       0.000      29.000  0.00000000
      16000    1000.000       0.000      29.000  0.00000000
      17000    1000.000       0.000      29.000  0.00000000
      18000    1000.000       0.000      28.596  0.00000000
      19000    1000.000       0.000      28.000  0.00000000
      20000    1000.000       0.000      28.000  0.00000000

```