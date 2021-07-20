import genetic
import logistics



def experiment_1():
    # population = 50; generations = 150; Pc = 0.6; Pm = 0.25
    solution = genetic.GeneticAlg()

    # solution.getAllFitness([16.21, 3.65, 0.78] )
    solution.runSolver()
    print(solution.chromosomes)

experiment_1()