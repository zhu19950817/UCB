# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:25:20 2017

@author: ZHU HANHUA
"""

import numpy as np
from matplotlib import pylab as plt


num_samples = 10000
num_arms = 3
num_Iteration = 100
train_data = [0.2,0.5,0.3]

# generate the train label.
def init():
    train_sample = np.tile(train_data, (num_samples,1)) # make a size of num_samples*num_arms matrix for matrix operation.
    train_label = np.random.rand(num_samples, len(train_data)) < train_sample # generate the train_label randomly. the size of train_label is num_samples*num_arms.
    return train_label

# uniformly random.
def random(evaluation_params):
    return np.random.randint(0, len(evaluation_params)) # choose an arm randomly.

# the UCB algorithm.
def UCB(evaluation_params):
    total = float(evaluation_params.sum()) # the total number of running.
    total_chosen = evaluation_params.sum(1) # the number of chosen. the size of total_chosen is arms*1.
    successes = evaluation_params[:,0] # the number of right choice. the size of successes is arms*1.
    means = successes / total_chosen # compute the means of right choice of each arms. the size of means is arms*1.
    variances = means - means**2 # compute the xj(t). the size of variances is arms*1.
    UCB = variances + np.sqrt(2 * np.log(total) / total_chosen) # get the UCB value of each arms. the size of UCB is arms*1.
    return np.argmax(UCB) # return the numeber of arm which has the maximum UCB value.

# an improved version of UCB algorithm.
def UCB_tuned(evaluation_params):
    total = float(evaluation_params.sum()) 
    total_chosen = evaluation_params.sum(1)
    successes = evaluation_params[:,0]
    means = successes / total_chosen
    variances = means - means**2
    UCB = means + np.sqrt(np.minimum(variances + np.sqrt(2*np.log(total) / total_chosen), 0.25 ) * np.log(total) / total_chosen) # the calculation formula of UCB value is changed.
    return np.argmax(UCB)

# the Multi-armed bandit algorithm.
def MBT(train_label, train_data, function):
    best_action = np.argmax(train_data)
    evaluation_params = np.zeros((num_arms, 2)) # a num_arms*2 matrix which used to save the number of righ choice and wrong choice.
    evaluation_params[:,0] += 1 # initial setting is 1.
    evaluation_params[:,1] += 1
    regret = np.zeros(num_samples)
    profit = np.zeros(num_samples) # a num_samples*1 size matrix used to count the profit of each choice.
    for i in range(0, num_samples):
        choice = function(evaluation_params) # get the choice of arms in this round.
        if train_label[i, choice] == 1: # compare to the train_label.
            update = 0; # if the choice is right, success++.
        else:
            update = 1;# if the choice is wrong, loss++.
        evaluation_params[choice, update] += 1 # update the right and wrong record.
        regret[i] = train_label[i, best_action] - train_data[choice]
        profit[i] = train_data[choice] # compute the profit of this round.
    ret = np.cumsum(profit) # accumulate the profit. the size of ret is num_samples*1.
    regret = np.cumsum(regret)
    regret = regret/num_samples
    return ret,regret



# the main code.
profit = np.zeros((num_samples, 3)) # a num_samples*3 matrix used to save the profit of three algorithm.
regret = np.zeros((num_samples, 3))
for i in range(0, num_Iteration): # iterate fixed times and take the average value of profit.
    train_label = init()
    p_u, r_u = MBT(train_label, train_data, random) # compute the profit of uniformly random.
    p_U, r_U = MBT(train_label, train_data, UCB) # compute the profit of UCB algorithm.
    p_t, r_t = MBT(train_label, train_data, UCB_tuned) # compute the profit of improved version of UCB algorithm.
    profit[:,0] += p_u
    profit[:,1] += p_U
    profit[:,2] += p_t
    regret[:,0] += r_u
    regret[:,1] += r_U
    regret[:,2] += r_t
# plot the graph.
print(regret[99,2])
plt.plot(profit/num_Iteration)
plt.title('The average profit of 3 algorithm under 100 times of interation')
plt.ylabel('profit')
plt.xlabel('Round Index')
plt.legend(('Random','UCB','UCB_tuned'),loc='lower right')
plt.savefig('The average profit')
plt.show()

plt.plot(regret/num_Iteration)
plt.title('The average regret of 3 algorithm under 100 times of interation')
plt.ylabel('regret')
plt.xlabel('Round Index')
plt.legend(('Random','UCB','UCB_tuned'),loc='lower right')
plt.savefig('The average regret')
plt.show()

ans = 0.05457