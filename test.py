from control.matlab import *
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import math
import copy
import logistics


chromo = [4.8, 5.23, 2.22]

def printCleanMatrix(array):
    for list in array:
        print(list)
    print("DONE \n")

kp = chromo[0]  # Random initial value between (2,18)
ti = chromo[1]  # Random initial value between (1.05, 9.42)
td = chromo[2]  # Random initial value between (0.26, 2.37)

# Compute G
g = kp * tf([ti * td, ti, 1], [ti, 0])

# Compute transfer function
f = tf(1, [1, 6, 11, 6, 0])

pid_sys = feedback(series(g, f), 1)
t = np.linspace(0, 50, 1000)
sysinfo = stepinfo(pid_sys)

i_s_e = 0
t_r = sysinfo['RiseTime']
t_s = sysinfo['SettlingTime']
m_p = sysinfo['Overshoot']

# t = np.linspace(0, 0.01, 100) 
repr = step(pid_sys,t)  # compute e(t)
# repr2 = step(pid_sys)
# printCleanMatrix(repr1[-1])

for i in range(len(repr[0])):
    i_s_e += (repr[0][i]) * (repr[0][i])

# print(t, len(repr1[0]))

print("ise ",i_s_e,  "tr ",t_r,"ts ", t_s, "mp",m_p)



    
