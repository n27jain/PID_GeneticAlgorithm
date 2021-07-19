from control.matlab import *
import scipy.integrate as integrate
import numpy as np


def integrand(x, e_function):
    # print("e_function", e_function)
    # print(e_function ** 2)
    return e_function ** 2


def runScore():
    kp = 10  # Random initial value between (2,18)
    ti = 5  # Random initial value between (1.05, 9.42)
    td = 1.5  # Random initial value between (0.26, 2.37)

    # Compute G
    g = kp * tf([ti * td, ti, 1], [ti, 0])

    # Compute transfer function
    f = tf(1, [1, 6, 11, 6, 0])

    pid_sys = feedback(series(g, f), 1)
    sysinfo = stepinfo(pid_sys)

    t = np.linspace(0, 0.01, 100)
    repr = step(pid_sys)  # compute e(t)
    
    i_s_e = 0
    t_r = sysinfo['RiseTime']
    t_s = sysinfo['SettlingTime']
    m_p = sysinfo['Overshoot']


    for number in repr[0]:
        i_s_e += (number-1) * (number-1)
    
    print("t_r: ", t_r)
    print("t_s: ", t_s)
    print("m_p: ", m_p)
    print("i_s_e: ", i_s_e)


    return None


runScore()