from operator import itemgetter

from numpy.core.fromnumeric import sort

matrix = [ [4,5,6,99], [1,2,3,0], [7,0,9,1]]

print(sorted(matrix, key=itemgetter(3)))