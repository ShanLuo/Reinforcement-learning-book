
import numpy as np 
import random
import matplotlib.pyplot as plt
import pdb

plot_color = ['r', 'g', 'b']
epsilon = [0, 0.01, 0.1]

num_play=1000
num_run=2000
num_arms = 10

#Reward distributions
for i in range(3):
	r_select = np.zeros((num_run,num_play))
	a_select = np.zeros((num_run,num_play), dtype=int)
	num_opt = np.zeros((num_run,num_play), dtype=int)

	for run in range(num_run):
		#Create 10 arms whose action values are a normal distribution
		#action values: mean = 0; variance 1
		mu, sigma = 0, 1		
		q_a = np.random.normal(mu, sigma, num_arms)

		sam_sum = np.zeros(num_arms)
		sam_count = np.zeros(num_arms, dtype=int)
		sam_avg = np.zeros(num_arms)

		R_t = np.zeros(num_arms)

		for t in range(num_play):
			#The actual reward R_t selected from a normal distribution
			#R_t: mean = q_a; variance 1
			for arm in range(num_arms):
				R_t[arm] = np.random.normal(q_a[arm], 1, 1)

			#With probability epsilon, choose the greedy action
			if np.random.uniform(0.0,1.0) < epsilon[i]:
				# exploration
				a_select[run][t] = random.randrange(num_arms)
			else:
				#exploitation
				a_select[run][t] = np.argmax(sam_avg)

		 #    #With probability epsilon, choose the greedy action
			# if t in random.sample(range(1000), int(epsilon[i]*1000)):
			# 	a_select[run][t] = random.randrange(num_arms)
			# #Otherwise, choose one from all the actions randomly
			# else:
			# 	a_select[run][t] = np.argmax(sam_avg)
			#pdb.set_trace()
			r_select[run][t] = R_t[a_select[run][t]]
			num_opt[run][t] = (a_select[run][t] == np.argmax(q_a))
			sam_sum[a_select[run][t]] += R_t[a_select[run][t]]
			sam_count[a_select[run][t]] +=1
			sam_avg[a_select[run][t]] = sam_sum[a_select[run][t]] / sam_count[a_select[run][t]]

	plt.figure(1)
	p1 = plt.plot(np.arange(num_play), np.mean(r_select, axis =0), plot_color[i], label = 'epsilon = %.2f' % (epsilon[i]))
	plt.legend(loc='lower right', framealpha=0.8)
	plt.figure(2)
	p2 = plt.plot(np.arange(num_play), np.sum(num_opt, axis =0)/num_run, plot_color[i], label = 'epsilon = %.2f' % (epsilon[i]))
	plt.legend(loc='lower right', framealpha=0.8)
plt.show()

