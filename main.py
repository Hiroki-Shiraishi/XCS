import numpy
import numpy.random
import pandas as pd
import xcs
import csv
import struct

"""
    An implementation of an 6-bit multiplexer problem for XCS
"""
#The maximum reward
rmax = 1000

#The number of steps we learn for
learning_steps = 1000

#The number of steps we validate for
validation_steps = 1000

#Load CSV dataset
file = './dataset/Mux-6.csv' 
with open(file,'r') as f: 
    reader = csv.reader(f)
    data = [low for low in reader]

"""
    Returns a random state of the multiplexer
"""
def state():
    return ''.join(data[numpy.random.randint(0,63)][i] for i in range(0,6))

"""
    The 6bit multiplexer is a single step problem, and thus always is at the end of the problem
"""
def eop():
    return True

"""
    Calculates the reward for performing the action in the given state
"""
def reward(state, action):
    address = state[:2]
    data = state[2:]

    #Check the action
    if str(action) == data[int(address, 2)]:
        return rmax
    else:
        return 0
 

"""
    Here is the main function. We'll train and validate XCS!!
"""

#Set parameters
parameters = xcs.Parameters()

#Construct an XCS instance
my_xcs = xcs.XCS(parameters, state, reward, eop)

#Train
print("The training has just began. The number of the train is 10,000. Wait for a couple of seconds, please...")
for j in range(learning_steps):
    my_xcs.run_experiment()
    if j % 1000 == 0:
        print("Iteration :", j)
    #my_xcs.print_population()
print("The Training is over. ")

#Make lists to generate CSV file
rewardList = [[0] for i in range(validation_steps)]
classifierList = [[0] * 10]
accuracyList = [[0] for i in range(validation_steps - 1000)]

#Validate
print("The next step is the validation. The number of the validation is 10,000. Please be patient...")
this_correct = 0
for j in range(validation_steps):
    rand_state = state()
    this_correct = this_correct + reward(rand_state, my_xcs.classify(rand_state))
    #print(my_xcs.classify(rand_state))
    #print(type(my_xcs.classify(rand_state)))
    if j % 1000 == 0 and j != 0:
        print("Iteration :", j, "AverageReward :", this_correct / (j+1))

    rewardList[j][0]  = reward(rand_state,my_xcs.classify(rand_state))
    if j == validation_steps - 1:
        classifierList[0][0] = "Classifier"
        classifierList[0][1] = "Condition"
        classifierList[0][2] = "Action"
        classifierList[0][3] = "Fitness"
        classifierList[0][4] = "Prediction"
        classifierList[0][5] = "Error"
        classifierList[0][6] = "Experience"
        classifierList[0][7] = "Time Stamp"
        classifierList[0][8] = "Action Set Size"
        classifierList[0][9] = "Numerosity"
        for clas in my_xcs.population:
            classifierList.append([clas.id, clas.condition, clas.action, clas.fitness, clas.prediction, clas.error, clas.experience, clas.time_stamp, clas.action_set_size, clas.numerosity])

print("ALL Performance " + ": " + str((this_correct / validation_steps / rmax) * 100) + "%");
print("The whole process is finished. After this, please check reward.csv, classifier.csv, and accuracy.csv files in 'result' folder. Thank you.")

#Make accuracy list (Percentage of correct answers per 1000 iterations)
ini_k = 0
for ini_k in range(validation_steps - 1000):
    sum_1000 = 0
    for k in range(ini_k, 1000 + ini_k):
        sum_1000 = sum_1000 + rewardList[k][0]
    accuracyList[ini_k][0] = sum_1000/1000
    #print(accuracyList)

#Make CSV files
with open('./result/reward.csv','w') as f:
    dataWriter = csv.writer(f, lineterminator='\n')
    dataWriter.writerows(rewardList)

with open('./result/classifier.csv', 'w') as f:
    dataWriter = csv.writer(f, lineterminator='\n')
    dataWriter.writerows(classifierList)

with open('./result/accuracy.csv', 'w') as f:
    dataWriter = csv.writer(f, lineterminator='\n')
    dataWriter.writerows(accuracyList)
