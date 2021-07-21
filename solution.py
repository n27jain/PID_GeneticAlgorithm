from control.matlab import *
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import math
import copy
import logistics

txt = None
def plotByPoints(points):
    x = []
    for i in range(len(points)):
        x.append(i)
    plt.xlabel('generation')
    # frequency label
    plt.ylabel('best fitness')
    plt.plot(x, points)
    plt.savefig('graph1.png')

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


    for i in range(len(repr[0])):
        i_s_e += (repr[0][i]) * (repr[0][i])
        if(repr[1][i] >= t_s):
            break

    total_cost = i_s_e +t_r+t_s+m_p

    if math.isnan(total_cost) or math.isnan(t_s):
        return -1
    return 1000/(1+total_cost)

def generateStart(population):
    sol_list = []
    for i in range(population):
        kp = logistics.customRand(2,18,2)
        ti = logistics.customRand(1.05,9.42,2)
        td = logistics.customRand(0.26,2.37,2)
        sol_list.append([kp,ti,td,0]) #add the 0 value for the fitness to be used and modified
    return sol_list

def russianRoulette(list,population):

    global txt
    sortedByfitness = copy.deepcopy(sorted(list, key = itemgetter(3)))
    newlist = []
    # add the 2 most fit chromosomes 
    newlist.append(copy.deepcopy(sortedByfitness[population-1]))
    newlist.append(copy.deepcopy(sortedByfitness[population-2]))
    rouletteRatio = []
    S = 0
    verify = 0

    for x in range (len(sortedByfitness)):
        if sortedByfitness[x][3] > 0:
            S += sortedByfitness[x][3]
    
    for i in range(population):
        if sortedByfitness[i][3] > 0: # solution is feasible 
            verify += sortedByfitness[i][3]/S
            rouletteRatio.append(verify)
        else:
            rouletteRatio.append(-1)
    
    
    
    for i in range(population - 2):
        check = logistics.customRand(0,0.99999,5) 
        for j in range(population):
            if rouletteRatio[j] >= check:
                newlist.append(copy.deepcopy((sortedByfitness[j])))
                break
        # print("newList: " , newlist)

    return newlist

def crossOver(list,population,pc):
    #cross over chromosomes 
    #exclude the 2 fittest solutions
    index_swap = []
    cloneList  = copy.deepcopy(list)
    for i in range(2,population):
        check = logistics.customRand(0,1,2)
        if check <= pc: #this chromosome needs to be crossed over 
            index_swap.append(i)
    if(len(index_swap) <= 1):
        return list
    for j in range(len(index_swap)):
        if j == len(index_swap)-1: # last element must cross over with the first 
            index_A = index_swap[j]
            index_B = index_swap[-1]
        else:
            index_A = index_swap[j]
            index_B = index_swap[j+1]
        
        crossOverPoint = int(logistics.customRand(1,2,0))
        if(crossOverPoint == 1):
            cloneList[index_A] = copy.deepcopy([cloneList[index_A][0],cloneList[index_B][1], cloneList[index_B][2],0])
        else:
            list[index_A] = copy.deepcopy([cloneList[index_A][0],cloneList[index_A][1], cloneList[index_B][2],0])
    return cloneList

def mutation(list,population,pm):
    global txt
    clone = copy.deepcopy(list)
    logistics.writeMatrix(clone,txt,"Matrix Before Mutation :")
    for i in range(2,population):
        logistics.writeMatrix(["0"],txt," Before clone"+"["+str(i)+ "] " + str(clone[i]))
        check = logistics.customRand(0,1,2)
        if check <= pm:
            # logistics.writeMatrix(["0"],txt,"Before clone[i][0] "+ str(clone[i]))
            
            clone[i][0] = copy.deepcopy(logistics.customRand(2,18,2))
            # logistics.writeMatrix(["0"],txt,"After clone[i][0] "+ str(clone[i]))

        check = logistics.customRand(0,1,2)
        if check <= pm:
            # logistics.writeMatrix(["0"],txt,"Before clone[i][0] "+ str(clone[i]))
            clone[i][1] = copy.deepcopy(logistics.customRand(1.05,9.42,2))
            # logistics.writeMatrix(["0"],txt,"After clone[i][0] "+ str(clone[i]))

        check = logistics.customRand(0,1,2)
        if check <= pm:
            clone[i][2] = copy.deepcopy(logistics.customRand(0.26,2.37,2))
        logistics.writeMatrix(clone,txt,"Matrix after Mutation : " + str(i))
    return clone


def main(population = 50, generations = 150, Pc = 0.6, Pm = 0.25):
    global txt
    plot_points = []
    surviors = generateStart(population)
    txt = logistics.createTxt("results.txt")
    for survior in surviors:
        fit = getFitness(survior)
        survior[3]= fit
    # logistics.writeMatrix( surviors,txt,"1.) Fitness :")

    surviors = russianRoulette(surviors,population)
    # logistics.writeMatrix(surviors,txt,"Roullete :")

    plot_points.append(surviors[-1][3]) #this point will come from the 

    surviors = crossOver(surviors,population,Pc)
    # logistics.writeMatrix(surviors,txt, "Crossover :")

    surviors = mutation(surviors,population,Pm)
    logistics.writeMatrix(surviors,txt,"mutation :", )
    
    # logistics.printCleanMatrix(surviors)
   
    last = None
    for i in range(generations):
        for survior in surviors:
            fit = getFitness(survior)
            survior[3] = fit
        logistics.writeMatrix(surviors,txt, str(i+2) + ".) Fitness :")
        if i != generations - 1:
            
            surviors = russianRoulette(surviors,population)
            logistics.writeMatrix( surviors,txt,"Roullete :")

            plot_points.append(surviors[-1][3])
            surviors = crossOver(surviors,population,Pc)
            logistics.writeMatrix( surviors,txt,"Crossover :")

            surviors = mutation(surviors, population,Pm)
            logistics.writeMatrix(surviors, txt,"mutations :")
            # logistics.printCleanMatrix(surviors)

    #get the best solution of the last iterations found in position 0
    sortedByfitness = sorted(surviors, key=itemgetter(3))
    print("BEST solution found :", sortedByfitness[-1])

    txt.close()
    return plot_points

def TEST1():
    # population = 50; generations = 150; Pc = 0.6; Pm = 0.25

    plotByPoints(main(generations=3))

# def see():
#     for i in range(10):
#         print(getFitness([2.96, 1.67, 1.26]))

# runScore()

# main(150)

TEST1()

# see()