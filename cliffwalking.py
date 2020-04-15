import numpy as np 
import random
import matplotlib.pyplot as plt
import pdb

#The grid world has a size of 4x12
height = 4
width = 12 
num_actions = 4
actions = [[1, 0], [-1, 0], [0, 1], [0,-1]]

start = [0, 0]  #x^
goal = [0, 11]  # |__> y
num_episodes = 500

gamma = 1
alpha = 0.1
epsilon = 0.1

num_runs = 10
#The sum of rewards for two algrithms
r_sum = np.zeros((num_runs, 2, num_episodes))

for run in range(num_runs):
	for i in range(2):
		q_table = np.zeros((height, width, num_actions))
		for episode in range(num_episodes):
			s_t = [0,0] #starting point

			while s_t != goal:
				#With probability epsilon, choose the greedy action
				if np.random.uniform(0.0,1.0) < epsilon:
					# exploration
					a_t = random.randrange(num_actions)
					#print('a_t-- ', a_t)
				else:
					#exploitation
					a_t = np.argmax(q_table[s_t[0]][s_t[1]][:])
					#print('a_t-- ', a_t)


				s_next = np.add(s_t, actions[a_t])
				s_next = np.ndarray.tolist(s_next)
				#pdb.set_trace()

				if (s_next[0] == 0) and (s_next[1] != 0) and (s_next[1] != 11):
					r = -100
					s_next = [0, 0]
				else:
					r = -1
					if s_next[0] < 0:
						s_next[0] = 0
					if s_next[1] < 0:
						s_next[1] = 0
					if s_next[0] > height - 1:
						s_next[0] = height - 1
					if s_next[1] > width -1:
						s_next[1] = width -1
				#SARSA
				if i == 0:
					if np.random.uniform(0.0,1.0) < epsilon:
						# exploration
						a_next = random.randrange(num_actions)
						#print('a_next-- ', a_next)
					else:
						#exploitation
						a_next = np.argmax(q_table[s_next[0]][s_next[1]][:])
						#print('a_next-- ', a_next)
					q_table[s_t[0]][s_t[1]][a_t] = q_table[s_t[0]][s_t[1]][a_t] + alpha * (r + gamma * q_table[s_next[0]][s_next[1]][a_next] - q_table[s_t[0]][s_t[1]][a_t])
					#print('s_t-- ',s_t, 'a_t-- ', a_t)
					
				else:
					#Q-learning
					q_table[s_t[0]][s_t[1]][a_t] = q_table[s_t[0]][s_t[1]][a_t] + alpha * (r + gamma * max(q_table[s_next[0]][s_next[1]][:]) - q_table[s_t[0]][s_t[1]][a_t])
					#a_next = np.argmax(q_table[s_next[0]][s_next[1]][:])
					#pdb.set_trace()

				r_sum[run][i][episode] += r
				s_t = s_next
				print(s_t, ' current episode: ',episode)

r_sum_avg = np.mean(r_sum, axis = 0)
r_sum_smooth = np.zeros((2, num_episodes))
for i in range(num_episodes):
	j = max(0,i-9)
	r_sum_smooth[0][i] = np.mean(r_sum_avg[0][j:i])
	r_sum_smooth[1][i] = np.mean(r_sum_avg[1][j:i])

plt.figure(1)
plt.plot(np.arange(num_episodes), r_sum_smooth[0][:], label='SARSA')
plt.plot(np.arange(num_episodes), r_sum_smooth[1][:], label='Q-learning')
plt.ylim([-100,0])
plt.legend(loc='lower right', framealpha=0.8)
plt.show()
