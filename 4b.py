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

def genRandomNum(start,end):
	deciNum = LCG()
	randoInt = round(start + deciNum * (end - start))
	return randoInt
count=0
"""with open('yearly_sales.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		print(row)
		line_count+=1
		if(line_count==5):
			break
	print(f'Processed {line_count} lines.')"""

popData=[] # for storing population data 
popVariance=0
noOfSamples = input("Enter the size of the sample\n") # the size of each sample 
noOfSamples = int(noOfSamples)
noOfTrials = input ("Enter the number of Trials\n") #No of sets of size {noOfSamples} need to be produced
noOfTrials = int(noOfTrials)


MeanOfSamples=[] # array to store the mean of each set of samples
VarianceOfSamples=[] # array to store the variance of each set of sampels
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




for trial in range(0,noOfTrials): #In each iteration of this loop a sample of size {noOfSamples} will be generated from the csv file and Its mean and variance will be calculated

	fileHandler = open("yearly_sales.csv",'r')
	randoNums = [] # To store random cust_ids whose data we are going to consider
	data= []  # array to store data corresponding each cust_id
	dataLength=0
	lineCount=0
	for i in range(0,noOfSamples): # generates {noOfSamples} random numbers between 100001 and 110000
		randoNums.append(genRandomNum(100001,110000))

	randoNums = np.array(randoNums)
	randoNums = np.sort(randoNums) # sorting random numbers to make finding their data easier

	randoCount=0

	for line in fileHandler: # finding the data corresponding every cust_id and storing them in data=[]
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

	sampleVariance = (1/(len(data)-1))*totalVal # sample variance 
	MeanOfSamples.append(sampleMean) # storing the mean of the current sample produced
	VarianceOfSamples.append(sampleVariance) # storing the variance of the current sample

	fileHandler.close()


MeanOfSamples = np.array(MeanOfSamples) 
MEAN = np.sum(MeanOfSamples)/len(MeanOfSamples) # Mean of all the sets of samples 
VarianceOfSamples = np.array(VarianceOfSamples) 
VARIANCE = np.sum(VarianceOfSamples/(len(VarianceOfSamples))) # variance of all the sets of samples
print("No of samples ",noOfSamples)
print("No of trials ",noOfTrials)
print(f"Sets of {noOfSamples} samples variance ",VARIANCE)
print("Mean of the samples is ",MEAN)


		




	



