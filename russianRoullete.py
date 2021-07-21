def russianRoulette(list,population):
  
    global txt

    sortedByfitness = copy.deepcopy(sorted(list, key = itemgetter(3)))
    newlist = []
    # add the 2 most fit chromosomes 
    newlist.append(sortedByfitness[population-1])
    newlist.append(sortedByfitness[population-2])

    print("newlist best:" , newlist)
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
    print("S:" , S, "verify: ", verify)

    for i in range(population -2):
        check = logistics.customRand(0,0.99999,5) 
        for j in range(population):
            if rouletteRatio[j] >= check:
                newList.append((sortedByfitness[j]))
                break
        print("newList: " , newlist)
    
    return newList