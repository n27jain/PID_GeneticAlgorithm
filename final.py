from control.matlab import *
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import math
import copy
import logistics


def plotExperiment(experiments):
    plt.figure(figsize=(40,20))
    # 3, 4
    i = 0
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
    plt.savefig("final" + '.png')
    # for i in range(len(points)):
    #     x.append(i)
    # plt.figure()
    # plt.title(name)
    # plt.xlabel('generation')
    # plt.ylabel('best fitness')
    # plt.plot(x, points)
    

def getFitness(chromo):
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
    # t = np.linspace(0, 0.01, 100)
    repr = step(pid_sys)  # compute e(t)
    
    i_s_e = 0
    t_r = sysinfo['RiseTime']
    t_s = sysinfo['SettlingTime']
    m_p = sysinfo['Overshoot']

    if math.isnan(t_s): # infeasible solution
        return -1

    for i in range(len(repr[0])):
        i_s_e += (repr[0][i]) * (repr[0][i])
    total_cost = i_s_e +t_r*1000+t_s*100+m_p*100

    if math.isnan(total_cost): #edge case throws error
        return -1
    return 10000/(1+total_cost)

def generateStart(population):
    sol_list = []
    for i in range(population):
        kp = logistics.customRand(2,18,2)
        ti = logistics.customRand(1.05,9.42,2)
        td = logistics.customRand(0.26,2.37,2)
        sol_list.append(copy.deepcopy([kp,ti,td,0])) #add the 0 value for the fitness to be used and modified
    return sol_list

def russianRoulette(list,population):
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
            rouletteRatio.append(copy.deepcopy(verify))
        else:
            rouletteRatio.append(copy.deepcopy(-1))
    
    
    
    for i in range(population - 2):
        check = logistics.customRand(0,0.99999,5) 
        for j in range(population):
            if rouletteRatio[j] >= check:
                newlist.append(copy.deepcopy((sortedByfitness[j])))
                break

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
            cloneList[index_A] = [copy.deepcopy(cloneList[index_A][0]),copy.deepcopy(cloneList[index_B][1]), copy.deepcopy( cloneList[index_B][2] ),0]
        else:
            cloneList[index_A] = [copy.deepcopy(cloneList[index_A][0]),copy.deepcopy(cloneList[index_A][1]), copy.deepcopy( cloneList[index_B][2] ),0]
    return cloneList

def mutation(list,population,pm):
    clone = copy.deepcopy(list)
    for i in range(2,population):
        
        check = logistics.customRand(0,1,2)
        if check <= pm:
            clone[i][0] = copy.deepcopy(logistics.customRand(2,18,2))
        check = logistics.customRand(0,1,2)
        if check <= pm:
            clone[i][1] = copy.deepcopy(logistics.customRand(1.05,9.42,2))
        check = logistics.customRand(0,1,2)
        if check <= pm:
            clone[i][2] = copy.deepcopy(logistics.customRand(0.26,2.37,2))
    return clone


def main(sol = None,population = 50, generations = 150, Pc = 0.6, Pm = 0.25, title = "example"):
    
    plot_points = []
    best_sol = []
    if(sol == None):
        surviors = generateStart(population)
    else:
        surviors = sol
    for survior in surviors:
        fit = getFitness(survior)
        survior[3]= fit
   
    # print("Fitness!")
    surviors = russianRoulette(surviors,population)
    # print("russianRoulette!")
    plot_points.append(surviors[0][3]) #this point will come from the 
    best_sol.append(surviors[0])

    surviors = crossOver(surviors,population,Pc)
    # print("crossOver!")
    surviors = mutation(surviors,population,Pm)
    # print("mutation!")
    # print("generations: " , generations)
    for i in range(generations):
        # print("here")
        for survior in surviors:
            fit = getFitness(survior)
            survior[3] = fit
        # print("Fitness!")
        if i != generations - 1:
            surviors = russianRoulette(surviors,population)
            # print("russianRoulette!")
            plot_points.append(surviors[0][3])
            best_sol.append(surviors[0])
            surviors = crossOver(surviors,population,Pc)
            # print("crossOver!")
            surviors = mutation(surviors, population,Pm)
            # print("mutation!")
        # print(i)
           

    #get the best solution of the last iterations found in position 0
    print(title + " BEST solution found :", surviors[0])
    return plot_points


def RunTests():
    print("Starting Tests: ")
    #Question 3
    starting = generateStart(population=50)
    # TEST1() And Question 3
    plotPoints = []
    # plotByPoints(main(sol = starting,population = 50, generations = 150, Pc = 0.6, Pm = 0.25),name="TEST1_BASE_CASE")
    plotPoints.append([main(sol = starting,population = 50, generations = 150, Pc = 0.6, Pm = 0.25, title="TEST_1"),"TEST1_BASE_CASE"])
    #TEST2()
    plotPoints.append([main(sol = starting , population = 50, generations = 200, Pc = 0.8, Pm = 0.25, title="TEST_2"),"TEST2_MORE_GENERATIONS"])
    
    
    #TEST3()
    plotPoints.append([main(sol = starting , population = 50, generations = 50, Pc = 0.6, Pm = 0.25, title="TEST_3"),"TEST3_LESS_GENERATIONS"])
# POPULATION SIZE
    #TEST4()
    plotPoints.append([main( population = 50, generations = 50, Pc = 0.6, Pm = 0.25, title="TEST_4"),"TEST4_POPULATION_BASE"])
    #TEST5()
    plotPoints.append([main( population = 25, generations = 50, Pc = 0.6, Pm = 0.25, title="TEST_5"),"TEST5_POPULATION_SMALL"])
    #TEST6()
    plotPoints.append([main(population = 400, generations = 50, Pc = 0.6, Pm = 0.25, title="TEST_6"),"TEST6_POPULATION_LARGE"])

# PROBABILITIES
    starting = generateStart(population=50)
    #TEST7()
    plotPoints.append([main(sol=starting, population = 50, generations = 80, Pc = 0.8, Pm = 0.25, title="TEST_7"),"TEST7_HIGH_PC"])
    #TEST8()
    plotPoints.append([main(sol=starting, population = 50, generations = 80, Pc = 0.4, Pm = 0.25, title="TEST_8"),"TEST8_LOW_PC"])
    #TEST9()
    plotPoints.append([main(sol=starting, population = 50, generations = 80, Pc = 0.6, Pm = 0.5, title="TEST_9"),"TEST9_HIGH_PM"])
    #TEST10()
    plotPoints.append([main(sol=starting, population = 50, generations = 80, Pc = 0.6, Pm = 0.1, title="TEST_10"),"TEST10_LOW_PM"])

    plotExperiment(plotPoints)

    print("Tests Complete")

RunTests()

# see()