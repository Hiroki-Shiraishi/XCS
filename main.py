import numpy
import numpy.random
import xcs
import csv

"""
    An implementation of an N-bit multiplexer problem for XCS
"""
#The maximum reward
rmax = 1000

#Load CSV dataset(6bit)
file = './dataset/Mux-6.csv' 
with open(file,'r') as f: 
    reader = csv.reader(f)
    data = [low for low in reader]

"""
    Returns a random state of the multiplexer
"""
def state():
    #return ''.join(data[numpy.random.randint(0,63)][i] for i in range(0,6))
    return ''.join('0' if numpy.random.rand() > 0.5 else '1' for i in range(0,parameters.bit + 2 ** parameters.bit))

"""
    The N-bit multiplexer is a single step problem, and thus always is at the end of the problem
"""
def eop():
    return True

"""
    Calculates the reward for performing the action in the given state
"""
def reward(state, action):
    address = state[:parameters.bit]
    data = state[parameters.bit:]

    #Check the action
    if str(action) == data[int(address, 2)]:
        return rmax
    else:
        return 0

def system_error(state, action, action_prediction):
    return abs(reward(state, action) - action_prediction)

def val_with_spaces(val):
    str_val = str(val)
    return ''.join(' ' for i in range(11 - len(str_val))) + str_val


"""
    Here is the main function. We'll train and validate XCS!!
"""

#Set parameters
parameters = xcs.Parameters()
print("[ XCS General Parameters ]")
print("            bit =", parameters.bit)
print(" Learning Steps =", parameters.learning_steps)
print("              N =", parameters.N)
print("           beta =", parameters.beta)
print("          alpha =", parameters.alpha)
print("      epsilon_0 =", parameters.e0)
print("             nu =", parameters.nu)
print("          gamma =", parameters.gamma)
print("       theta_GA =", parameters.theta_GA)
print("            chi =", parameters.chi)
print("             mu =", parameters.mu)
print("      theta_del =", parameters.theta_del)
print("          delta =", parameters.delta)
print("      theta_sub =", parameters.theta_sub)
print("            P_# =", parameters.p_hash)
print("            p_I =", parameters.p_I)
print("      epsilon_I =", parameters.e_I)
print("            F_I =", parameters.F_I)
print("        p_explr =", parameters.p_explr)
print("doGAsubsumption =", parameters.do_GA_subsumption)
print("doASSubsumption =", parameters.do_action_set_subsumption)
print("crossoverMethod = two-point\n")

print("[ XCS Optional Settings]")
print("            tau =", parameters.tau)

#Construct an XCS instance
my_xcs = xcs.XCS(parameters, state, reward, eop)

#Make lists to generate CSV
rewardList = [[0] for i in range(parameters.learning_steps)]
classifierList = [[0] * 10]
accuracyList = [[0] for i in range(parameters.learning_steps - 1000)]

#Begin learning and validation
this_correct = this_syserr = this_population_size = this_covering_occur_num= 0
print("\n  Iteration      Reward      SysErr     PopSize  CovOccRate")
print("=========== =========== =========== =========== ===========")
for j in range(parameters.learning_steps):
    #Learning(Pure Exploration)
    my_xcs.run_experiment()

    #Validation(Pure Exploitation)
    rand_state = state()
    this_correct += reward(rand_state, my_xcs.exploitation(rand_state))
    this_syserr += system_error(rand_state, my_xcs.exploitation(rand_state), my_xcs.action_prediction(rand_state, my_xcs.exploitation(rand_state)))
    this_population_size += len(my_xcs.population)
    this_covering_occur_num += my_xcs.covering_occur_num
    s = parameters.summary_interval

    if (j+1) % s == 0:
         print(val_with_spaces(j+1),
               val_with_spaces(f'{this_correct / s:.3f}'),
               val_with_spaces(f'{this_syserr / s:.3f}'),
               val_with_spaces(f'{this_population_size / s:.3f}'),
               val_with_spaces(f'{this_covering_occur_num / s / 100:.8f}'))

         this_correct = this_syserr = this_population_size = this_covering_occur_num = my_xcs.covering_occur_num = 0

    rewardList[j][0] = reward(rand_state, my_xcs.exploitation(rand_state))
    if j == parameters.learning_steps - 1:
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

print("The whole process is over. After this, please check reward.csv, classifier.csv, and accuracy.csv files in 'result' folder. Thank you.")

#Make accuracy list (Percentage of correct answers per 1000 iterations)
for ini_k in range(parameters.learning_steps - 1000):
    sum_1000 = 0
    for k in range(ini_k, 1000 + ini_k):
        sum_1000 += rewardList[k][0]
    accuracyList[ini_k][0] = sum_1000/1000

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