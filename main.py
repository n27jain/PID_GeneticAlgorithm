from control.matlab import *
import scipy.integrate as integrate
import numpy as np
import random
from numpy.core.fromnumeric import sort
from operator import itemgetter
import math
import copy

def pop_matrix_column(matrix, column_num):
    # print(matrix)
    for i in range(len(matrix)):
        # print("HERE", matrix[i], '/n')
        try:
            if(len(matrix[i]) >= column_num+1):  
                matrix[i].pop(column_num)
        except:
            print("failed at: ", i , len(matrix))
            print(matrix[i])

def customRand(start,end,place):
    return round(random.uniform(start, end),place)

def getFitness(chromo):
    kp = chromo[0]  # Random initial value between (2,18)
    ti = chromo[1]  # Random initial value between (1.05, 9.42)
    td = chromo[2]  # Random initial value between (0.26, 2.37)

    # Compute G
    g = kp * tf([ti * td, ti, 1], [ti, 0])

    # Compute transfer function
    f = tf(1, [1, 6, 11, 6, 0])

    pid_sys = feedback(series(g, f), 1)
    sysinfo = stepinfo(pid_sys)
    # t = np.linspace(0, 0.01, 100)
    repr = step(pid_sys)  # compute e(t)
    
    i_s_e = 0
    t_r = sysinfo['RiseTime']
    t_s = sysinfo['SettlingTime']
    m_p = sysinfo['Overshoot']

    for number in repr[0]:
        i_s_e += (number-1) * (number-1)
    total_cost = i_s_e +t_r+t_s+m_p
    if math.isnan(total_cost):
        return -1
    return 1/(1+total_cost)

def generateStart(population):
    # Kp – (2,18)
    # Ti – (1.05,9.42)
    # Td – (0.26, 2.37)
    sol_list = []
    for i in range(population):
        kp = customRand(2,18,2)
        ti = customRand(1.05,9.42,2)
        td = customRand(0.26,2.37,2)
        sol_list.append([kp,ti,td])
    return sol_list

def russianRoulette(list,population):
    # of the 50 chromosomes
    # select the top 2 fitest scores as chromosomes 1 and 2
    # for the remaing 48 we select from the 50 in the population 
    # using the roulette strategy
    sortedByfitness = sorted(list, key=itemgetter(3))
    surviors = []

    # add the 2 most fit chromosomes 
    surviors.append(sortedByfitness[-1])
    surviors.append(sortedByfitness[-2])

    rouletteRatio = []
    S = 0
    for sol in  sortedByfitness:
        if sol[3] > 0:
            S += sol[3]
    verify = 0
    for i in range(len(sortedByfitness)):
        if sortedByfitness[i][3] > 0:
            verify += sortedByfitness[i][3]/S
            rouletteRatio.append(verify)
        else:
            rouletteRatio.append(-1)
        
    # print("verify : ", verify)
    # print("rouletteRatio : ",rouletteRatio)
    for i in range(population - 2):
        # lets keep it significant by 5 decimal places since there are many possible ties at 
        # 4 decimal places
        # we should also not keep 1 as a possibly randomly generated number. 
        # from experimentation the sum of the ratios is arround 0.9999999999999996
        check = customRand(0,0.99999,5) 
        for i in range(len(sortedByfitness)):
            if rouletteRatio[i] >= check:
                surviors.append(sortedByfitness[i])
                break
    pop_matrix_column(surviors, 4-1)
    # print(surviors)
    return surviors

def crossOver(list,population,pc):
    #cross over chromosomes 
    #exclude the 2 fittest solutions
    index_swap = []
    cloneList  = copy.deepcopy(list)
    for i in range(2,population):
        check = customRand(0,1,2)
        if check <= pc: #this chromosome needs to be crossed over 
            index_swap.append(i)
    # print("index_swap: ", index_swap, "/n")
    if(len(index_swap) <= 1):
        return list
    for j in range(len(index_swap)):
        if j == len(index_swap)-1: # last element must cross over with the first 
            index_A = index_swap[j]
            index_B = index_swap[-1]
        else:
            index_A = index_swap[j]
            index_B = index_swap[j+1]
        
        crossOverPoint = int(customRand(1,2,0))
        # print("INDEX: " , list[index_A], list[index_B], "CROSSPOINT: ", crossOverPoint)
        if(crossOverPoint == 1):
            cloneList[index_A] = [cloneList[index_A][0],cloneList[index_B][1], cloneList[index_B][2]]
        else:
            list[index_A] = [cloneList[index_A][0],cloneList[index_A][1], cloneList[index_B][2]]
        # print("change : " , cloneList[index_A])
    # for val in index_swap:
    #     print("original: ", list[val], "/n")
    #     print("new: ", cloneList[val], "/n")
    return cloneList



def main():
    surviors = generateStart(population=50)
    for survior in surviors:
        fit = getFitness(survior)
        # print("survior,fit : ", survior, fit )
        survior.append(fit)

    surviors = russianRoulette(surviors,50)
    surviors = crossOver(surviors,population=50,pc=0.6)
    

    
def test():
    chrome = [12.82, 2.64, 0.32]
    fit = getFitness(chrome)
    print(fit)


# runScore()

main()