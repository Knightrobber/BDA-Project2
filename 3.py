from random import randint


switchCount=0 #Counts the number of times you win upon switching the door
switchNotCount=0 # Counts the number of times you win upon Not switching the door
trials = 10000

for i in range(0,trials):
	winDoor = randint(1,3)
	chosenDoor = randint(1,3)
	winSwitch=0
	winNotSwitch=0
	if winDoor==chosenDoor : # you win if you don't switch
		winNotSwitch=1
		switchNotCount+=1
	else:					 # you win if you switch
		winSwitch=1
		switchCount+=1


print(f"Prob of winning if you switch the door ",switchCount/trials)
print(f"Prob of winning if dot't switch the door ",switchNotCount/trials)

