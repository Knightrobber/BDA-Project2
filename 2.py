from random import random
from random import randint
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
#U gay
mean1 = [10,20] # set the mean of x and y coordinates
sigma1 = [2,2] # set the sigma along x and y
xCor=[] # array to store randomly generated x - coordinates
yCor=[] # array to store randomly generated xy- coordinates
occurances=[] # array to store the occurances of each point
randoLength=0
noOfDataPoints = 50 # The number of distinct data data points to be produced
sampleSize = 400 # The total sample size, which is multiplied with the probablity of each point to find its occurances



def fx(X,Y,code): # function to compute the value of multi-variate normal function
	if(code ==1):
		part1 = 1/(2*math.pi*sigma1[0]) #here co-variance matrix is sigma^2I and both sigma are same
		part2 = (-1/2)*(1/math.pow(sigma1[0],2) ) * (  math.pow(X-mean1[0],2) + math.pow(Y-mean1[1],2)  )

		fxVal = part1 * math.exp(part2)
		return fxVal
	if(code==2):
		part1 = 1/(2*math.pi*sigma2[0]) #here co-variance matrix is sigma^2I and both sigma are same
		part2 = (-1/2)*(1/math.pow(sigma2[0],2) ) * (  math.pow(X-mean2[0],2) + math.pow(Y-mean2[1],2)  )

		fxVal = part1 * math.exp(part2)
		return fxVal

def makeMeanAndSigma(mean,sigma): # used to create a random  mean and sigma for a new set of data points so that we can create a linearly seperable dataset
	# If 2 sets of data need to be classifyable, their means shouldn't overlap
	meanProduced = [0,0]
	Case = randint(1,2)
	sig = randint(abs(sigma[0]-2*sigma[0]),sigma[0]+2*sigma[0])
	sigmaProduced = [sig,sig]
	if Case==1:#mean2[0] is variable
		SubCase = randint(1,2)
		if(SubCase==1): #mean2[1]<mean1[1]
			meanProduced[1] = mean[1] - 2*sigma[0] - 2*sigmaProduced[0] - sigma[0]
			#print("mean2[0] is variable with mean2[1] lesss")
		if(SubCase ==2):#mean2[1]>mean1[1]
			meanProduced[1] = mean[1] + 2*sigma[0] + 2*sigmaProduced[0] + sigma[0]
			#print("mean2[0] is variable with mean2[1] great> ")
		tempMean = [mean[0]-2*mean[0],mean[0]+2*mean[0]]
		tempMean = np.array(tempMean)
		minMean = min(tempMean)
		maxMean = max(tempMean)
		meanProduced[0] = randint(minMean,maxMean)
	
	
	if Case==2 :#mean2[1] is variable
		SubCase = randint(1,2)
		if(SubCase==1): #mean2[0]<mean1[0]
			meanProduced[0] = mean[0] - 2*sigma[0] - 2*sigmaProduced[0] - sigma[0]
			#print("mean2[1] is variable with mean2[0] lesss")
		if(SubCase ==2):#mean2[1]>mean1[1]
			meanProduced[0] = mean[0] + 2*sigma[0] + 2*sigmaProduced[0] + sigma[0]
			#print("mean2[1] is variable with mean2[0] great")
		tempMean = [mean[1]-2*mean[1],mean[1]+2*mean[1]]
		tempMean = np.array(tempMean)
		minMean = min(tempMean)
		maxMean = max(tempMean)
		meanProduced[1] = randint(minMean,maxMean)
	return meanProduced,sigmaProduced
		





while randoLength<noOfDataPoints: # genearating random points for the initially set mean and sigma
# all the random data points are generated between 2 SD of the mean as 98% of the data points fall within this limit
	tempxCor = round( ( mean1[0] - 2*sigma1[0] ) +  (random()*4*sigma1[0]) ,2)
	tempyCor = round( ( mean1[1] - 2*sigma1[1] ) +  (random()*4*sigma1[1]) ,2)

	if((tempxCor not in xCor) or (tempyCor not in yCor)):
		xCor.append(tempxCor)
		yCor.append(tempyCor)
		prob = fx(tempxCor,tempyCor,1)
		frequency = round(prob*sampleSize)
		occurances.append(frequency)
		randoLength+=1

plt.scatter(xCor,yCor,marker="*",color="green") # plotting the first set of data points





mean2,sigma2 = makeMeanAndSigma(mean1,sigma1)
print(f"The Preset mean {mean1}, The preset Sigma {sigma1}")
print(f"Computer generated mean {mean2}  Sigma  {sigma2} for other dataset")
xCor2=[]
yCor2=[]
occurances2=[]
randoLength=0

while randoLength<noOfDataPoints:

	tempxCor = round( ( mean2[0] - 2*sigma2[0] ) +  (random()*4*sigma2[0]) ,2)
	tempyCor = round( ( mean2[1] - 2*sigma2[1] ) +  (random()*4*sigma2[1]) ,2)

	if((tempxCor not in xCor) or (tempyCor not in yCor)):
		xCor2.append(tempxCor)
		yCor2.append(tempyCor)
		prob = fx(tempxCor,tempyCor,2)
		frequency = round(prob*sampleSize)
		occurances2.append(frequency)
		randoLength+=1


plt.scatter(xCor2,yCor2,marker="*",color="red") #plotting the next of data points




#PLA
# ax >0 for class 1
# ax <0 for class 2

convergence = 0 # becomes 1 when all data has been perfectly classified
a = [1,1,1] # starting value of the weights

while not convergence:
	for i in range(0,len(xCor)): # iterating through every point in both datasets
		
		val = round(a[0]*1  +a[1]*xCor[i] + a[2]*yCor[i],2) # taking the first point from the first dataset
		if val<0: # if the value is <0 then weights are updated and the loop is restarted again
			a[0] += 1
			a[1] += xCor[i]
			a[2] += yCor[i]
			break

		#val for points in second dataset are calculated by multiplying the data points by -1
		val = round(a[0]*-1 + a[1]* -xCor2[i] + a[2]*-yCor2[i],2) # taking the first point from the second dataset
		if val<0: # 
			a[0] += -1
			a[1] += -xCor2[i]
			a[2] += -yCor2[i]
			break
		if i == len(xCor)-1:# if there was no val<0, the data points have been correctly classified 
			convergence = 1
			print("Converged")
	
print("The weights are ",a) 

# Plotting the line
lineX = []
lineY = []
tempx1 = np.array(xCor)
tempy1 = np.array(yCor)
tempx2 = np.array(xCor2)
tempy2 = np.array(yCor2)
yMin = [min(tempy1),min(tempy2)]
yMin = min(yMin)
yMax = [max(tempy1),max(tempy2)]
yMax = max(yMax)
for i in range(round(yMin),round(yMax)):
	lineY.append(i)
	x = (a[0] + i*a[2])/-a[1]
	lineX.append(x)
plt.plot(lineX,lineY)



# Plotting the PDF's of both the data points
#Here I am plotting the histograms of both the data points 
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

xpos = xCor
ypos = yCor
num_elements = len(xpos)
zpos = np.zeros(len(xCor))
dx = np.ones(len(xCor))
dy = np.ones(len(xCor))
dz = occurances

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')

xpos2 = xCor2
ypos2 = yCor2
num_elements2 = len(xpos2)
zpos2 = np.zeros(len(xCor2))
dx2 = np.ones(len(xCor2))
dy2 = np.ones(len(xCor2))
dz2 = occurances2

ax1.bar3d(xpos2, ypos2, zpos2, dx2, dy2, dz2, color='red')
plt.xlabel("X")
plt.ylabel("Y")



#Here a 3d mesh of the ppoints is beign plotted
"""XTemp = xCor
for i in range(0,len(xCor2)):
	XTemp.append(xCor2[i])

YTemp = yCor
for i in range(0,len(yCor2)):
	YTemp.append(yCor2[i])

occurTemp = occurances
for i in range(0,len(occurances2)):
	occurTemp.append(occurances2[i])"""


fig = go.Figure(data=[go.Mesh3d(x=xCor2, y=yCor2, z=occurances2 ,color='lightpink', opacity=1)])

fig.add_trace(go.Mesh3d(x=xCor, y=yCor, z=occurances, color='cyan', opacity=0.5))



fig.show()


plt.show()

		




