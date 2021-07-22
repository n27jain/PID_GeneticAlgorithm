from control.matlab import *
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import math
import copy
import logistics

experiments = []
experiments.append([[1,2,3,4,324,23432,2343,0], "first"])
experiments.append([[1,2,3,4,324,23432,2343,0], "secodn"])
experiments.append([[1,2,3,4,324,23432,2343,0], "t"])
experiments.append([[1,2,3,4,324,23432,2343,0], "for"])
experiments.append([[1,2,3,4,324,23432,2343,0], "fif"])
experiments.append([[1,2,3,4,324,23432,2343,0], "six"])
experiments.append([[1,2,3,4,324,23432,2343,0], "sev"])
experiments.append([[1,2,3,4,324,23432,2343,0], "eig"])
experiments.append([[1,2,3,4,324,23432,2343,0], "nin"])
experiments.append([[1,2,3,4,324,23432,2343,0], "ten"])
plt.figure(figsize=(40,20))
for i in range(len(experiments)):
    x = []
    experiment = experiments[i]
    y = experiment[0]
    title = experiment[1]
    for j in range(len(y)):
        x.append(j+1)
    plt.subplot(3, 4, i+1)
    plt.plot(x,y)
    plt.title(experiment[1])
    plt.xlabel('generation')
    plt.ylabel('best fitness')
plt.savefig("experiment_results_complete" + '.png')