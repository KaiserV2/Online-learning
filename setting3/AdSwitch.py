import numpy as np
import random
import matplotlib.pyplot as plt

T = 20000
nbArms = 10 # arms 
C1 = 16.1

lower = 0
amplitude = 20

delta_t = 400 # trick to 
delta_s = 100

t = 0


def with_prob(prob):
	return random.random() <= prob

def mean_(l):
	return np.mean(l) if len(l) else 0.0

def find_max_i(gap):
	if gap < 0:
		I = 1
	elif np.isclose(gap, 0):
		I = 1
	else:
		I = max(1, int(np.floor(4 - np.log2(gap))))
	return I

def n_s_t(arm, s, t):
	all_rewards_of_arm = all_rewards[arm]
	all_rewards_s_to_t = [r for (tau, r) in all_rewards_of_arm.items() if s <= tau <= t]
	return len(all_rewards_s_to_t)

def mu_hat_s_t(arm, s, t):
	all_rewards_of_arm = all_rewards[arm]
	all_rewards_s_to_t = [r for (tau, r) in all_rewards_of_arm.items() if s <= tau <= t]
	return mean_(all_rewards_s_to_t)

def check_good_arms_changes():
	for good_arm in GOOD_set:
		for s1 in range(t_l, t+1, delta_s):
			for s2 in range(s1, t+1, delta_s):
				for s in range(t_l, t+1, delta_s):
					n_s1_s2_a = n_s_t(good_arm, s1, s2)
					mu_hat_s1_s2_a = mu_hat_s_t(good_arm, s1, s2)
					n_s_t_a = n_s_t(good_arm, s, t)
					mu_hat_s_t_a = mu_hat_s_t(good_arm, s, t)
					abs_diff = abs(mu_hat_s1_s2_a - mu_hat_s_t_a)
					confidence_radius_s1s2 = np.sqrt(2 * max(1, np.log(T)) / max(1, n_s1_s2_a))
					confidence_radius_st = np.sqrt(2 * max(1, np.log(T)) / max(1, n_s_t_a))
					# check condition (3)
					if abs_diff > confidence_radius_s1s2 + confidence_radius_st:
						return True
	return False

def check_bad_arms_changes():
	for bad_arm in BAD_set:
		for s in range(t_l, t+1, delta_s):
			n_s_t_a = n_s_t(bad_arm, s, t)
			mu_hat_s_t_a = mu_hat_s_t(bad_arm, s, t)
			abs_diff = abs(mu_hat_s_t_a - mu_tilde_of_l[bad_arm])
			confidence_radius_st = np.sqrt(2 * max(1, np.log(T))/ max(n_s_t_a, 1))
			gap = gap_Delta_tilde_of_l[bad_arm] / 4
			# check condition (4)
			if abs_diff > gap + confidence_radius_st:
				return True
	return False


if __name__ == "__main__":
	# sth may be from the input
	

	# input rewards from file
	lower = 10000000000
	upper = 0
	all_rewards_input = [{} for _ in range(nbArms)]
	with open("dataset2.txt", "r") as f:
		lines = f.readlines()
		for t in range(len(lines)):
			line = lines[t].split(' ')
			for i in range(nbArms):
				lower = min(lower, float(line[i]))
				upper = max(upper, float(line[i]))
				# all_rewards_input[i][t+1] = (float(line[i]) - lower) / (upper - lower)
				all_rewards_input[i][t+1] = float(line[i])

	print(lower, upper)

	# normalize
	for t in range(T):
		for i in range(nbArms):
			all_rewards_input[i][t+1] = (upper - all_rewards_input[i][t+1]) / (upper - lower)


	GOOD_set = set(range(nbArms))
	BAD_set = set()
	set_S = [set() for i in range(nbArms)]
	mu_tilde_of_l = np.zeros(nbArms, dtype = float)
	gap_Delta_tilde_of_l = np.zeros(nbArms, dtype = float)

	all_rewards = [{} for i in range(nbArms)]
	history_of_plays = []


	# start the game
	l = 0 # episode
	t = 0
	t_l = 0
	total_reward = 0
	average_cumulative_rewards = []

	final_results = []
	chosen_arms = []

	# main part of the game
	while (t < T):
		# Start a new episode
		l = l + 1
		t_l = t + 1
		GOOD_set = set(range(nbArms)) # GOOD_{t+1}
		BAD_set = set() # BAD_{t+1}
		history_of_plays = [] 
		total_reward = 0
		average_cumulative_rewards = []
		# Next time step:
		while (t < T):
			t = t + 1
			print("t = ", t)
			# 1. Add checks for bad arms:
			for bad_arm in BAD_set:
				gap_Delta_tilde_of_l_a = gap_Delta_tilde_of_l[bad_arm]
				max_i = find_max_i(gap_Delta_tilde_of_l_a) + 1
				for i in range(1, max_i):
					probability_to_add = 2 ** (-i) * np.sqrt(l / (nbArms * T * np.log(T)))
					if with_prob(probability_to_add):
						triplet = (2**(-i), np.ceil(2**(2*i+1) * np.log(T)), t)
						set_S[bad_arm].add(triplet)
			
			# 2. Select an arm:
			taus = [float("+inf") for arm in range(nbArms)]
			for arm in GOOD_set | {a for a in range(nbArms) if set_S[a]}:
				look_in_past = 1
				while look_in_past < len(history_of_plays) and history_of_plays[-look_in_past] != arm:
					look_in_past += 1
				taus[arm] = t - look_in_past

			chosen_arm = np.argmin(taus)
			history_of_plays.append(chosen_arm)

			# print("history_of_plays =", history_of_plays)
			# print("taus =", taus)


			# Receive reward r_t from input
			total_reward += upper - (upper - lower) * all_rewards_input[chosen_arm][t]
			all_rewards[chosen_arm][t] = all_rewards_input[chosen_arm][t]
			average_cumulative_rewards.append(total_reward / (t - t_l + 1))

			# debug
			# print("chosen_arm = ", chosen_arm)
			# print("total_reward / t = ", total_reward / (t - t_l + 1))
			
			should_start_new_episode = False
			# 3. Check for changes of good arms:
			if t % delta_t == 0:
				if not should_start_new_episode:
					should_start_new_episode = check_good_arms_changes()

			if should_start_new_episode:
				final_results.append(average_cumulative_rewards)
				chosen_arms.append(history_of_plays)
				break

			# 4. Check for changes of bad arms:
			if t % delta_t == 0:
				if not should_start_new_episode:
					should_start_new_episode = check_bad_arms_changes()

			if should_start_new_episode:
				final_results.append(average_cumulative_rewards)
				chosen_arms.append(history_of_plays)
				break

			# 4'. Recompute S_{t+1}
			for bad_arm in BAD_set:
				new_set = set()
				for triplet in set_S[bad_arm]:
					_, n, s = triplet
					n_s_t_a = n_s_t(bad_arm, s, t)
					if n_s_t_a < n:
						new_set.add(triplet)
				set_S[bad_arm] = new_set

			# 5. Evict arms from GOOD_t:
			if t % delta_t == 0:
				GOOD_set_new = GOOD_set.copy()
				BAD_set_new = BAD_set.copy()
				mu_hat_s_t_a_potential = {}
				gap_Delta_potentail = {}
				for good_arm in GOOD_set:
					# check condition (1)
					for s in range(t_l, t+1, delta_s):
						mu_hat_s_t_a = mu_hat_s_t(good_arm, s, t)
						# print("good_arm =", good_arm, "mu_hat_s_t_a =", mu_hat_s_t_a)
						'''
						mu_hat_s_t_good = []
						for other_arm in GOOD_set:
							print("other_arm =", other_arm)
							mu_hat_s_t_good.append(mu_hat_s_t(other_arm, s, t))
							print("mu_hat_s_t_good[other_arm] =", mu_hat_s_t(other_arm, s, t))
						'''
						mu_hat_s_t_good = [mu_hat_s_t(other_arm, s, t) for other_arm in GOOD_set]
						mu_hat_s_t_best = max(mu_hat_s_t_good)
						# print("mu_hat_s_t_best =", mu_hat_s_t_best)
						n_s_t_a = n_s_t(good_arm, s, t)
						print("n_s_t_a =", n_s_t_a, "s=", s, "t=", t)
						if n_s_t_a < 2:
							continue
						gap_Delta = mu_hat_s_t_best - mu_hat_s_t_a
						gap_to_check = np.sqrt(C1 * max(1, np.log(T)) / max(1, n_s_t_a - 1))
						
						print("gap_Delta =", gap_Delta, "gap_to_check =", gap_to_check)
						print("chosen_arm = ", chosen_arm)
						print("total_reward / t = ", total_reward / (t - t_l + 1))
						
						if gap_Delta > gap_to_check:
							print("detect a bad_arm:", good_arm)
							BAD_set_new.add(good_arm)
							mu_hat_s_t_a_potential[good_arm] = mu_hat_s_t_a
							gap_Delta_potentail[good_arm] = gap_Delta
							break
				BAD_diff = BAD_set_new.difference(BAD_set)
				for evicted_arm in BAD_diff:
					mu_tilde_of_l[evicted_arm] = mu_hat_s_t_a_potential[evicted_arm]
					gap_Delta_tilde_of_l[evicted_arm] = gap_Delta_potentail[evicted_arm]
					set_S[evicted_arm] = set()

				print("BAD_set = ", BAD_set)
				print("BAD_set_new = ", BAD_set_new)

				BAD_set = BAD_set_new
				GOOD_set = set(range(nbArms)).difference(BAD_set)

			# TODO: where to put it? 
			if should_start_new_episode:
				final_results.append(average_cumulative_rewards)
				chosen_arms.append(history_of_plays)
				break

	print("episode = ", l)
	print()
	for i in range(l):
		print(i+1)
		print(chosen_arms[i])
		print(final_results[i])
		print()

	# plot
	# average_cumulative_rewards = [-x for x in average_cumulative_rewards]
	# plt.plot(range(1, T+1), average_cumulative_rewards, color = 'k')
	# plt.title("average cumulative rewards of AdSwitch")
	# plt.xlabel("time")
	# plt.ylabel("average cumulative rewards of AdSwitch")
	# plt.show()


