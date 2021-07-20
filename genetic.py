from control.matlab import *
import numpy as np
from numpy.core.fromnumeric import sort
from operator import itemgetter
import math
import copy
import logistics

class GeneticAlg:
    def __init__(self,population=50,generations = 150, pc = 0.6, pm =0.25):
        self.population = population
        self.generations = generations
        self.pc = pc
        self.pm = pm
        self.chromosomes = []

    def getAllFitness(self):
        for i in range(len(self.chromosomes)):
            kp = self.chromosomes[i][0] 
            ti = self.chromosomes[i][1] 
            td = self.chromosomes[i][2]

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
            try:
                if math.isnan(total_cost):
                    self.chromosomes[i][2] = -1;
                else:
                    self.chromosomes[i][2] = 1/(1+total_cost)
            except: 
                print("FOUND ERROR", self.chromosomes[i], i )
    
    def generateStart(self):
        list = []
        for i in range(self.population):
            kp = logistics.customRand(2,18,2)
            ti = logistics.customRand(1.05,9.42,2)
            td = logistics.customRand(0.26,2.37,2)
            list.append([kp,ti,td,0])
        self.chromosomes = list
            
    def russianRoulette(self):
        # of the 50 chromosomes
        # select the top 2 fitest scores as chromosomes 1 and 2
        # for the remaing 48 we select from the 50 in the population 
        # using the roulette strategy
        logistics.printCleanMatrix(self.chromosomes)
        sortedByfitness = sorted(self.chromosomes, key=itemgetter(3))
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
        
        for i in range(self.population - 2):
            # lets keep it significant by 5 decimal places since there are many possible ties at 
            # 4 decimal places
            # we should also not keep 1 as a possibly randomly generated number. 
            # from experimentation the sum of the ratios is arround 0.9999999999999996
            check = logistics.customRand(0,0.99999,5) 
            for i in range(len(sortedByfitness)):
                if rouletteRatio[i] >= check:
                    surviors.append(sortedByfitness[i])
                    break
        self.chromosomes = surviors

    def crossOver(self):
        #cross over chromosomes 
        #exclude the 2 fittest solutions

        logistics.printCleanMatrix(self.chromosomes)

        index_swap = []
        cloneList  = copy.deepcopy(self.chromosomes)
        for i in range(2,self.population):
            check = logistics.customRand(0,1,2)
            if check <= self.pc: #this chromosome needs to be crossed over 
                index_swap.append(i)
        if(len(index_swap) <= 1): # only one chromosome to cross over
            #hence do nothing and let the solutions remain the same
            return 
        for j in range(len(index_swap)):
            if j == len(index_swap)-1: # last element must cross over with the first 
                index_A = index_swap[j]
                index_B = index_swap[-1]
            else:
                index_A = index_swap[j]
                index_B = index_swap[j+1]
            
            crossOverPoint = int(logistics.customRand(1,2,0))
            if(crossOverPoint == 1):
                cloneList[index_A] = [cloneList[index_A][0],cloneList[index_B][1], cloneList[index_B][2]]
            else:
                cloneList[index_A] = [cloneList[index_A][0],cloneList[index_A][1], cloneList[index_B][2]]
        self.chromosomes = cloneList

    def mutation(self):
        clone = copy.deepcopy(self.chromosomes)
        for i in range(2,self.population):
            check = logistics.customRand(0,1,2)
            if check <= self.pm:
                clone[i][0] = logistics.customRand(2,18,2)
            check = logistics.customRand(0,1,2)
            if check <= self.pm:
                clone[i][1] = logistics.customRand(1.05,9.42,2)
            check = logistics.customRand(0,1,2)
            if check <= self.pm:
                clone[i][2] = logistics.customRand(0.26,2.37,2)
        return clone
    def runSolver(self):
        self.generateStart() #create 50 random solutions to begin with
        for i in range(self.generations):
            self.getAllFitness() #update the array with the fitness value 
            self.russianRoulette()#sort chromosomes by fitness and select surviors (first 2 are the best fit)
            self.crossOver() #cross-over the the surviors 
            self.mutation() #mutate the surviors 