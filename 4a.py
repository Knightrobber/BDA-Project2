from time import time
import csv
import numpy as np
import math

seed = time()
StateI=seed
StateF=0
def LCG():
	a = 8989542241
	c = 95231
	m = 2 ** 64
	global StateI,StateF
	StateF = (a*StateI + c) % m
	StateI = StateF
	return StateF/(m-1)

def genRandomNum(start,end): #returns a random number between start and end value
	deciNum = LCG()
	randoInt = round(start + deciNum * (end - start))
	return randoInt

popData=[] # for storing population data 
popVariance=0
popMean=0 # to store mean of population
lineCount=0
fileHandler = open("yearly_sales.csv",'r')
for line in fileHandler:
	if len(line)!=1 and lineCount!=0:
		tempLine = line
		broke = tempLine.split(",")
		popData.append(float(broke[1])) # stroring each value corresponding to every Cust_Id
	lineCount+=1

popData = np.array(popData)  # creating numpy array out of the population data
popMean = np.sum(popData)/len(popData)
totalVal = 0 # to store each (xi - Xmean)^2
for i in range(0,len(popData)):
	currentVal = popData[i]
	diff = math.pow( currentVal - popMean, 2 )
	totalVal+=diff
popVariance = (1/(len(popData)-1))*totalVal

print("Population mean ",popMean," Population Variance ",popVariance)
fileHandler.close()


sampleSize = input("Enter the sample size\n")
sampleSize = int(sampleSize)
randoNums = []
data=[]
for i in range(0,sampleSize): # generates {sampleSize} random numbers between 100001 and 110000
	randoNums.append(genRandomNum(100001,110000))

randoNums = np.array(randoNums)
randoNums = np.sort(randoNums) # sorting random numbers to make finding their data easier

randoCount=0
fileHandler = open("yearly_sales.csv",'r')
lineCount = 0
for line in fileHandler: # finding the data corresponding every random cust_id and storing them in data=[]
	if(randoCount>=len(randoNums)):
		break
	if len(line)!=1 and lineCount!=0:
		broke = line.split(",")
		if int(broke[0])==randoNums[randoCount]:
			data.append(float(broke[1]))
			randoCount+=1
			if randoCount<len(randoNums):
				if randoNums[randoCount] == int(broke[0]):
					while randoNums[randoCount] == int(broke[0]):
						data.append(float(broke[1]))
						randoCount+=1
						if(randoCount>=len(randoNums)):
							break
	lineCount+=1

sampleMean = np.sum(data)/len(data) # sample mean
	
totalVal=0 # to store each (xi - Xmean)^2
for i in range(0,len(data)):
	currentVal = data[i]
	diff = math.pow( currentVal - sampleMean, 2 )
	totalVal+=diff

sampleVariance = (1/(len(data)-1))*totalVal 

print("Sample mean ",sampleMean)
print("Sample variance",sampleVariance)

fileHandler.close()