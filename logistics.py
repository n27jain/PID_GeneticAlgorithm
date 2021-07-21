from os import wait3
import random

file = open("results.txt", mode='w')
def printCleanMatrix(array):
    for list in array:
        print(list)
    print("DONE \n")
def popMatrixCol(matrix, col):
    for i in range(len(matrix)):
        try:
            if(len(matrix[i]) >= col+1):  
                matrix[i].pop(col)
        except:
            print("failed at: ", i , len(matrix))
            print(matrix[i])
    

def customRand(start,end,place):
    return round(random.uniform(start, end),place)

def createTxt(name):
    return open(name, mode='w')

def writeMatrix(array,file, title = "example"):
    file.write(title + "\n")
    for list in array:
        file.write(str(list) + "\n")
    
    
    file.write(str(list) + "\n" + "\n" + "\n" + "\n" + "\n")