import math
import matplotlib.pyplot as plt
import random
import numpy as np


p = input("Enter the value of P\n")
p = float(p)
#arrays for storing coordinates
x =[]
y =[]
xNve =[]
yNve = []
noOfPoints = 10
interval =1000
limit =1

for i in range(-noOfPoints,noOfPoints): # for values of x in range (-10.000,10.000) we calculate the corresponding y value and plot it
	
	fx = 1 - abs(i)**p
	if fx<0:# as the value of mod y can't be less than 0, we ingnore that value of x and y
		continue

	for j in range(0,interval): #fx is poisitve, so we break i which is an interger into 1000 decimal no between i,i-1
		inter = 1/interval
		val = i + j*inter
		fx = limit - abs(val)**p
		if(fx<0):
			break
		
		x.append(val) # the value of x is appended to both arrays as every value of x has 2 distinct y values
		xNve.append(val)
		fx = fx**(1/p)	
		y.append(fx) # when we remove mod from y, we get 2 answers for y +,-
		yNve.append(-fx)

xTemp = x
yTemp = y
for i in range(0,len(xNve)): # plotting the graph
	xTemp.append(xNve[i])
	yTemp.append(yNve[i])
plt.plot(xTemp,yTemp,color="blue")
plt.fill_between(xTemp,yTemp,0,color="blue") # filling the graph to show complete area
plt.title(f"P = {p}")
plt.show()







