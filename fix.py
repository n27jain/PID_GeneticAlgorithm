import control as co
import numpy as np
import matplotlib.pyplot as plt


solution = [200,105,26] #Kp,Ti,Td


def transferFunction(chromo):
    numerator =[chromo[1]/100*chromo[2]/100,chromo[1]/100,1]
    denominator = [chromo[1]/100]
    G = chromo[0] * co.tf(numerator,denominator)
    return G

chromo = [200,105,26] #Kp,Ti,Td
Kp= chromo[0]/100
Ti = chromo[1]/100
Td = chromo[2]/100

# numerator =[chromo[1]/100*chromo[2]/100,chromo[1]/100,1]
# denominator = [chromo[1]/100,0]
# G = co.tf(numerator,denominator)
# t = np.linspace(0,10,1000)
# t1, y1 = co.step_response(G,t)

s = co.tf('s')
G = (1+Ti*Td*s+Ti*s)/(Ti*s)
print(G)



# plt.plot(t1,y1)
# plt.grid()
