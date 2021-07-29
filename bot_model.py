#!/usr/bin/env python
# coding: utf-8

# Reproducing model in paper with Xiaodan: https://www.overleaf.com/project/5c375cf8f540a47e999968db
# 
# Notes:
# * Need Python 3.6 or later; eg: `module load python/3.6.6`
# * remember link direction is following, opposite of info spread!
# 

# In[ ]:


import os
import networkx as nx
import random
import numpy
import numpy as np
import math
import statistics
import csv
import matplotlib.pyplot as plt
from scipy import stats
from operator import itemgetter
from collections import defaultdict
import sys
import fcntl
import time
import pickle


# In[ ]:


# parameters and utility globals

n_humans = 1000 # 10k for paper
beta = 0.1 # bots/humans ratio; 0.1 for paper
p = 0.5 # for network clustering; 0.5 for paper
k_out = 3 # average no. friends within humans & bots; 3 for paper
alpha = 15 # depth of feed; 15 for paper
mu = 0.75 # average prob of new meme vs retweet; 0.75 for paper or draw from empirical distribution
# phi = 1 
# gamma = 0.1 
epsilon = 0.01 # threshold used to check for steady-state convergence
n_runs = 10 # number of simulations to average results
cvsfile = 'results.csv' # to save results for plotting

phis = [1, 5, 10] # bot deception >= 1: meme fitness higher than quality 
gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0] # infiltration: probability that a human follows each bot

# if called with gamma as a command line params
if len(sys.argv) == 2:
  gamma = float(sys.argv[1])
  assert(0 <= gamma <= 1)


# In[ ]:


# create a network with random-walk growth model

def random_walk_network(net_size):
  if net_size <= k_out + 1: # if super small just return a clique
    return nx.complete_graph(net_size, create_using=nx.DiGraph())
  G = nx.complete_graph(k_out, create_using=nx.DiGraph()) 
  for n in range(k_out, net_size):
    target = random.choice(list(G.nodes()))
    friends = [target]
    n_random_friends = 0
    for _ in range(k_out - 1):
      if random.random() < p:
        n_random_friends += 1
    friends.extend(random.sample(list(G.successors(target)), n_random_friends))
    friends.extend(random.sample(list(G.nodes()), k_out - 1 - n_random_friends))
    G.add_node(n)
    for f in friends:
      G.add_edge(n, f)
  return G


# In[ ]:


# sample a bunch of objects from a list without replacement 
# and with given weights (to be used as probabilities), which can be zero
# NB: cannot use random_choices, which samples with replacement
#     nor numpy.random.choice, which can only use non-zero probabilities

def sample_with_prob_without_replacement(elements, sample_size, weights): 
  
  # first remove the elements with zero prob, normalize rest
  assert(len(elements) == len(weights))
  total = 0
  non_zeros = []
  probs = []
  zeros = []
  for i in range(len(elements)):
    if weights[i] > 0:
      non_zeros.append(elements[i])
      probs.append(weights[i])
      total += weights[i]
    else: 
      zeros.append(elements[i])
  probs = [w/total for w in probs]

  # if we have enough elements with non-zero probabilities, sample from those
  if sample_size <= len(non_zeros):
    return numpy.random.choice(non_zeros, p=probs, size=sample_size, replace=False)
  else:
    # if we need more, take all the elements with non-zero probability
    # plus a random sample of the elements with zero probability
    return non_zeros + random.sample(zeros, sample_size - len(non_zeros))


# In[ ]:


# create network of humans and bots
# preferential_targeting is a flag; if False, random targeting

def init_net(preferential_targeting):

  # humans
  H = random_walk_network(n_humans)
  for h in H.nodes:
    H.nodes[h]['bot'] = False

  # bots
  n_bots = int(n_humans * beta) 
  B = random_walk_network(n_bots)
  for b in B.nodes:
    B.nodes[b]['bot'] = True

  # merge and add feed
  # feed is array of (quality, fitness) tuples
  G = nx.disjoint_union(H, B)
  assert(G.number_of_nodes() == n_humans + n_bots)
  humans = []
  bots = []
  for n in G.nodes:
    G.nodes[n]['feed'] = []
    if G.nodes[n]['bot']:
      bots.append(n)
    else:
      humans.append(n)

  # humans follow bots
  w = [G.in_degree(h) for h in humans]
  for b in bots:
    n_followers = 0
    for _ in humans:
      if random.random() < gamma:
        n_followers += 1
    if preferential_targeting:
      followers = sample_with_prob_without_replacement(humans, n_followers, w)
    else:
      followers = random.sample(humans, n_followers)
    for f in followers:
      G.add_edge(f, b)

  return G


# In[ ]:


# return (quality, fitness) tuple depending on bot flag
# using https://en.wikipedia.org/wiki/Inverse_transform_sampling

def get_meme(bot_flag):

  if bot_flag:
    exponent = 1 + (1 / phi)
  else:
    exponent = 1 + phi
  u = random.random()
  fitness = 1 - (1 - u)**(1 / exponent)
  if bot_flag:
    quality = 0
  else:
    quality = fitness
  return (quality, fitness)


# In[ ]:


# count the number of forgotten memes as a function of in_degree (followers)
# using a global variable that needs to be reset

forgotten_memes = {}
def forgotten_memes_per_degree(n_forgotten, followers):
  if followers in forgotten_memes:
    forgotten_memes[followers] += n_forgotten
  else:
    forgotten_memes[followers] = n_forgotten


# In[ ]:


# track retweet memes
# using a global variable that needs to be reset

tracked_memes = {}
def track_memes(meme):
  if meme in tracked_memes:
    tracked_memes[meme] += 1
  else:
    tracked_memes[meme] = 1


# In[ ]:


# count bad meme selected times
# using a global variable that needs to be reset

bad_memes_seleted_time = defaultdict(lambda :[0, 0]) # {"meme": [human_node_select, bot_node_select]}
def select_time(meme, bot_flag):
  if bot_flag:
    bad_memes_seleted_time[meme][1] += 1
  else:
    bad_memes_seleted_time[meme][0] += 1


# In[ ]:


# a single simulation step in which one agent is activated

def simulation_step(G,
                    count_forgotten_memes=False,
                    track_retweet_meme=False,
                    count_select_time=False):
  agent = random.choice(list(G.nodes()))
  memes_in_feed = G.nodes[agent]['feed']
  if len(memes_in_feed) and random.random() > mu:
    # retweet a meme from feed selected on basis of its fitness
    fitnesses = [m[1] for m in memes_in_feed]
    meme = random.choices(memes_in_feed, weights=fitnesses, k=1)[0]
  else:
    # new meme
    meme = get_meme(G.nodes[agent]['bot'])
  
  if track_retweet_meme:
    track_memes(meme)
  
  if count_select_time and meme[0] == 0:
    select_time(meme, G.nodes[agent]['bot'])

  # spread (truncate feeds at max len alpha)
  followers = G.predecessors(agent)
  for f in followers:
    #print('follower feed before:', ["{0:.2f}".format(round(m[0], 2)) for m in G.nodes[f]['feed']])   
    G.nodes[f]['feed'].insert(0, meme)
    if len(G.nodes[f]['feed']) > alpha:
      if count_forgotten_memes and G.nodes[f]['bot'] == False:
        # count only forgotten memes with zero quality
        forgotten_zeros = 0
        for m in G.nodes[f]['feed'][alpha:]:
          if m[0] == 0:
            forgotten_zeros += 1
        forgotten_memes_per_degree(forgotten_zeros, G.in_degree(f))
      del G.nodes[f]['feed'][alpha:]
      #print('follower feed after :', ["{0:.2f}".format(round(m[0], 2)) for m in G.nodes[f]['feed']]) 
  #print('Bot' if G.nodes[agent]['bot'] else 'Human', 'posted', meme, 'to', G.in_degree(agent), 'followers', flush=True) 


# In[ ]:


# calculate average quality of memes in system

def measure_average_quality(G, count_bot=False):
  total = 0
  count = 0
  for agent in G.nodes:
    if count_bot == True or G.nodes[agent]['bot'] == False:
      for m in G.nodes[agent]['feed']:
        count += 1
        total += m[0]
  return total / count


# In[ ]:


# calculate fraction of low-quality memes in system

def measure_average_zero_fraction(G):
  count = 0
  zeros = 0 
  for agent in G.nodes:
    if G.nodes[agent]['bot'] == False:
      for m in G.nodes[agent]['feed']:
        count += 1
        if m[0] == 0: 
          zeros += 1 
  return zeros / count


# In[ ]:


# new network from old but replace feed with average quality

def add_avq_to_net(G):
  newG = G.copy()
  for agent in newG.nodes:
    if len(newG.nodes[agent]['feed']) < 1:
      print('Bot' if newG.nodes[agent]['bot'] else 'Human', 'has empty feed')
    newG.nodes[agent]['feed'] = float(statistics.mean([m[0] for m in G.nodes[agent]['feed']]))
  return newG


# In[ ]:


# main simulation 
# steady state is determined by small relative change in average quality
# returns average quality at steady state 

def simulation(preferential_targeting_flag, return_net=False,
               count_forgotten=False,
               track_meme=False,
               count_select=False):
  network = init_net(preferential_targeting_flag)
  n_agents = nx.number_of_nodes(network)
  old_quality = 100
  new_quality = 200
  time_steps = 0
  while max(old_quality, new_quality) > 0 and abs(new_quality - old_quality) / max(old_quality, new_quality) > epsilon: 
    #print('time_steps = ', time_steps, ', q = ', new_quality) 
    time_steps += 1
    for _ in range(n_agents):
      simulation_step(network,
                      count_forgotten_memes=count_forgotten,
                      track_retweet_meme=track_meme,
                      count_select_time=count_select)
  
    old_quality = new_quality
    new_quality = measure_average_quality(network)
  if return_net:
    return (new_quality, network)
  else:
    return new_quality


# In[ ]:


# append to file, locking file and waiting if busy in case of multi-processing

def save_csv(data_array): 
  with open(cvsfile, 'a', newline='') as file:
    while True:
      try:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        writer = csv.writer(file)
        writer.writerow(data_array)
        fcntl.flock(file, fcntl.LOCK_UN)
        break
      except:
        time.sleep(0.1)


# In[ ]:


# read from file

def read_csv(filename):
  q_mean_random = {} 
  q_stderr_random = {} 
  q_mean_preferential = {} 
  q_stderr_preferential = {} 
  q_mean_ratio = {} 
  q_stderr_ratio = {} 
  with open(filename, newline='') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
      q_mean_random[row[0]] = row[1]
      q_stderr_random[row[0]] = row[2]
      q_mean_preferential[row[0]] = row[3]
      q_stderr_preferential[row[0]] = row[4]
      q_mean_ratio[row[0]] = row[5]
      q_stderr_ratio[row[0]] = row[6]
  return(q_mean_random, q_stderr_random, 
         q_mean_preferential, q_stderr_preferential, 
         q_mean_ratio, q_stderr_ratio)


# In[ ]:


base = 1.5
def logbase(x):
    return np.log(x)/np.log(base)

def get_count(list):
    count = {}
    for i, q in enumerate(list):
        if q in count.keys():
            count[q]+=1
        else:
            count[q]=1
    return count

def get_distr(count):
    distr = {}
    sum = 0
    aver = 0
    for a in count.keys():
        sum += count[a]
        aver += a*count[a]
        bin = int(logbase(a))
        if bin in distr.keys():
            distr[bin] += count[a]
        else:
            distr[bin] = count[a]
    return distr, sum

def getbins(distr, sum):
    mids = []
    heights = []
    bin = sorted(distr.keys())
    for i in bin:
        start = base ** i
        width = base ** (i+1)-start
        mid = start + width/2
        mids.append(mid)
        heights.append(distr[i]/(sum * width))
    return mids, heights


# In[ ]:


def draw_heatmap(ax, data, xticks, yticks, xlabel, ylabel, cmap, title, vmax=None, vmin=None):
    data = data[::-1, :]
    if vmin == None:
        vmin = data[0][0]
        for i in data:
            for j in i:
                if j<vmin:
                    vmin=j
    if vmax == None:
        vmax = data[0][0]
        for i in data:
            for j in i:
                if j>vmax:
                    vmax=j

    map = ax.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    yticks = yticks[::-1]
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks, fontsize=14)
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, fontsize=14)#, rotation=40
    cb = plt.colorbar(mappable=map, cax=None, ax=None)
    cb.ax.tick_params(labelsize=12)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title)


# Above are definitions
# 
# ---
# 
# Below is main experiment

# In[ ]:


# experiment, save results to CSV file
# this is slow for large n_humans; better to run in parallel 
# on a server or cluster, eg, one process per gamma value
save_dir = "results/random"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

q_random_all = {}
for phi in phis:
  for gamma in gammas:
    q_random = []
    valid_tracked_memes_random_all = []
    bad_memes_selected_time_random_all = {}
    avg_quality_random_all = []
    avg_diversity_random_all = []
    for sim in range(n_runs):
      print('Running Simulation ', sim, ' for phi = ', phi, ', gamma = ', gamma, ' ...', flush=True)
    
      # reset global variable
      global forgotten_memes, tracked_memes, bad_memes_seleted_time
      assert(forgotten_memes != None)
      assert(tracked_memes != None)
      assert(bad_memes_seleted_time != None)
      forgotten_memes = {}
      tracked_memes = {}
      bad_memes_seleted_time = defaultdict(lambda :[0, 0])

      # simulation start
      qr, qr_net = simulation(False, True, True, True, True) # random attach
      q_random.append(qr)
      if (phi, gamma) not in q_random_all:
        q_random_all[(phi, gamma)] = []
      q_random_all[(phi, gamma)].append(qr)
    
      #### statistic current nth-run data ####
      ## tracked meme ##
      valid_tracked_memes = []
      for meme in tracked_memes:
        valid = True
        for agent in qr_net.nodes:
          for m in qr_net.nodes[agent]['feed']:
            if meme == m:
              valid = False
        if valid:
          valid_tracked_memes.append((meme[0], tracked_memes[meme]))
      valid_tracked_memes_random_all.extend(valid_tracked_memes)
      ## end tracked meme ##
    
      ## bad meme select ##
      for meme, selected_time in bad_memes_seleted_time.items():
        if meme[1] not in bad_memes_selected_time_all:
          bad_memes_selected_time_random_all[meme[1]] = [0, 0]
        bad_memes_selected_time_random_all[meme[1]][0] += selected_time[0]
        bad_memes_selected_time_random_all[meme[1]][1] += selected_time[1]
      ## end bad meme select ##

      ## avg quality ##
      avg_quality_random_all.append(qr)
      ## end avg quality ##

      ## avg diversity ##
      for agent in qr_net.nodes:
        qualities = []
        fitnesses = []
        for m in qr_net.nodes[agent]['feed']:
          qualities.append(m[0])
          fitnesses.append(m[1])
        unique_qua, unique_qua_cnt = np.unique(qualities, return_counts=True)
        portion_of_qua = unique_qua_cnt / np.sum(unique_qua_cnt)
        diversity = - np.sum(portion_of_qua * np.log(portion_of_qua))
        avg_diversity_random_all.append(diversity)
        
        # unique_fit, unique_fit_cnt = np.unique(fitnesses, return_counts=True)
        # portion_of_fit = unique_fit_cnt / np.sum(unique_fit_cnt)
        # diversity = - np.sum(portion_of_fit * np.log(portion_of_fit))
        # avg_diversity_random_all.append(diversity)
      ## end avg diversity ##
      #### end statistic current nth-run data ####

    for fitness, selected_time in bad_memes_selected_time_all.items():
      bad_memes_selected_time_all[fitness][0] /= n_runs
      bad_memes_selected_time_all[fitness][1] /= n_runs

    # save tracked memes
    fp = open("{}/tracked_memes_random_phi{}_gamma{}.pkl".format(save_dir, phi, gamma), "wb")
    pickle.dump(valid_tracked_memes_random_all, fp)
    fp.close()

    # save bad meme selected times
    fp = open("{}/bad_memes_selected_time_random_phi{}_gamma{}.pkl".format(save_dir, phi, gamma), "wb")
    pickle.dump(bad_memes_selected_time_random_all, fp)
    fp.close()

    # save avg_quality
    fp = open("{}/avg_quality_random_phi{}_gamma{}.pkl".format(save_dir, phi, gamma), "wb")
    pickle.dump(np.mean(avg_quality_random_all), fp)
    fp.close()
    
    # save avg_fitness
    fp = open("{}/avg_diversity_random_phi{}_gamma{}.pkl".format(save_dir, phi, gamma), "wb")
    pickle.dump(np.mean(avg_diversity_random_all), fp)
    fp.close()

    # save kendall
    quality, number_selected = zip(*valid_tracked_memes_random_all)
    kendall_tau, _ = stats.kendalltau(quality, number_selected)
    fp = open("{}/kendall_random_phi{}_gamma{}.pkl".format(save_dir, phi, gamma), "wb")
    pickle.dump(kendall_tau, fp)
    fp.close()


# In[ ]:


save_dir = "results/prefer"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

q_prefer_all = {}
for phi in phis:
  for gamma in gammas:
    q_prefer = []
    valid_tracked_memes_prefer_all = []
    bad_memes_selected_time_prefer_all = {}
    avg_quality_prefer_all = []
    avg_diversity_prefer_all = []
    for sim in range(n_runs):
      print('Running Simulation ', sim, ' for phi = ', phi, ', gamma = ', gamma, ' ...', flush=True)

      # reset global variable
      global forgotten_memes, tracked_memes, bad_memes_seleted_time
      assert(forgotten_memes != None)
      assert(tracked_memes != None)
      assert(bad_memes_seleted_time != None)
      forgotten_memes = {}
      tracked_memes = {}
      bad_memes_seleted_time = defaultdict(lambda :[0, 0])

      # simulation start
      qp, qp_net = simulation(True, True, True, True, True) # preferential attach
      q_prefer.append(qp)
      if (phi, gamma) not in q_prefer_all:
        q_prefer_all[(phi, gamma)] = []
      q_prefer_all[(phi, gamma)].append(qp)
    
      #### statistic current nth-run data ####
      ## tracked meme ##
      valid_tracked_memes = []
      for meme in tracked_memes:
        valid = True
        for agent in qr_net.nodes:
          for m in qr_net.nodes[agent]['feed']:
            if meme == m:
              valid = False
        if valid:
          valid_tracked_memes.append((meme[0], tracked_memes[meme]))
      valid_tracked_memes_prefer_all.extend(valid_tracked_memes)
      ## end tracked meme ##
    
      ## bad meme select ##
      for meme, selected_time in bad_memes_seleted_time.items():
        if meme[1] not in bad_memes_selected_time_all:
          bad_memes_selected_time_prefer_all[meme[1]] = [0, 0]
        bad_memes_selected_time_prefer_all[meme[1]][0] += selected_time[0]
        bad_memes_selected_time_prefer_all[meme[1]][1] += selected_time[1]
      ## end bad meme select ##

      ## avg quality ##
      avg_quality_prefer_all.append(qp)
      ## end avg quality ##

      ## avg diversity ##
      for agent in qp_net.nodes:
        qualities = []
        fitnesses = []
        for m in qp_net.nodes[agent]['feed']:
          qualities.append(m[0])
          fitnesses.append(m[1])
        unique_qua, unique_qua_cnt = np.unique(qualities, return_counts=True)
        portion_of_qua = unique_qua_cnt / np.sum(unique_qua_cnt)
        diversity = - np.sum(portion_of_qua * np.log(portion_of_qua))
        avg_diversity_prefer_all.append(diversity)
        
        # unique_fit, unique_fit_cnt = np.unique(fitnesses, return_counts=True)
        # portion_of_fit = unique_fit_cnt / np.sum(unique_fit_cnt)
        # diversity = - np.sum(portion_of_fit * np.log(portion_of_fit))
        # avg_diversity_prefer_all.append(diversity)
      ## end avg diversity ##
      #### end statistic current nth-run data ####

    for fitness, selected_time in bad_memes_selected_time_all.items():
      bad_memes_selected_time_all[fitness][0] /= n_runs
      bad_memes_selected_time_all[fitness][1] /= n_runs

    # save tracked memes
    fp = open("{}/tracked_memes_prefer_phi{}_gamma{}.pkl".format(save_dir, phi, gamma), "wb")
    pickle.dump(valid_tracked_memes_prefer_all, fp)
    fp.close()

    # save bad meme selected times
    fp = open("{}/bad_memes_selected_time_prefer_phi{}_gamma{}.pkl".format(save_dir, phi, gamma), "wb")
    pickle.dump(bad_memes_selected_time_prefer_all, fp)
    fp.close()

    # save avg_quality
    fp = open("{}/avg_quality_prefer_phi{}_gamma{}.pkl".format(save_dir, phi, gamma), "wb")
    pickle.dump(np.mean(avg_quality_prefer_all), fp)
    fp.close()
    
    # save avg_fitness
    fp = open("{}/avg_diversity_prefer_phi{}_gamma{}.pkl".format(save_dir, phi, gamma), "wb")
    pickle.dump(np.mean(avg_diversity_prefer_all), fp)
    fp.close()

    # save kendall
    quality, number_selected = zip(*valid_tracked_memes_prefer_all)
    kendall_tau, _ = stats.kendalltau(quality, number_selected)
    fp = open("{}/kendall_prefer_phi{}_gamma{}.pkl".format(save_dir, phi, gamma), "wb")
    pickle.dump(kendall_tau, fp)
    fp.close()


# In[ ]:


q_ratio = []

for phi in phis:
  for gamma in gammas:
    q_random = q_random_all[(phi, gamma)]
    q_preferential = q_prefer_all[(phi, gamma)]
    q_ratio = (np.array(q_preferential) / np.array(q_random)).tolist()
    # save results to CSV file
    save_csv([gamma, statistics.mean(q_random), 
            statistics.stdev(q_random) / math.sqrt(n_runs), 
            statistics.mean(q_preferential), 
            statistics.stdev(q_preferential) / math.sqrt(n_runs), 
            statistics.mean(q_ratio), 
            statistics.stdev(q_ratio) / math.sqrt(n_runs)])


# In[ ]:


# plot data from CSV file

q_mean_random, q_stderr_random, q_mean_preferential, q_stderr_preferential, q_mean_ratio, q_stderr_ratio = read_csv(cvsfile)

ymin = [q_mean_ratio[x] - q_stderr_ratio[x] for x in q_mean_ratio.keys()]
ymax = [q_mean_ratio[x] + q_stderr_ratio[x] for x in q_mean_ratio.keys()]
plt.xlabel(r'$\gamma$', fontsize=16)
plt.ylabel('Average Quality Ratio', fontsize=16)
plt.xscale('log')
plt.axhline(y=1, lw=0.5, color='black')
plt.plot(list(q_mean_ratio.keys()), list(q_mean_ratio.values()))
plt.fill_between(list(q_mean_ratio.keys()), ymax, ymin, alpha=0.2)


# In[ ]:


# plot from files for different values of mu

plt.subplots(figsize=plt.figaspect(1.5))

_, _, _, _, ratio_mu75, stderr_mu75 = read_csv('results.csv')
ymin_mu75 = [ratio_mu75[x] - 2*stderr_mu75[x] for x in ratio_mu75.keys()]
ymax_mu75 = [ratio_mu75[x] + 2*stderr_mu75[x] for x in ratio_mu75.keys()]
plt.plot(list(ratio_mu75.keys()), list(ratio_mu75.values()), label=r'$\mu=0.75$')
plt.fill_between(list(ratio_mu75.keys()), ymax_mu75, ymin_mu75, alpha=0.2)

_, _, _, _, ratio_mu25, stderr_mu25 = read_csv('results.csv')
ymin_mu25 = [ratio_mu25[x] - 2*stderr_mu25[x] for x in ratio_mu25.keys()]
ymax_mu25 = [ratio_mu25[x] + 2*stderr_mu25[x] for x in ratio_mu25.keys()]
plt.plot(list(ratio_mu25.keys()), list(ratio_mu25.values()), label=r'$\mu=0.25$')
plt.fill_between(list(ratio_mu25.keys()), ymax_mu25, ymin_mu25, alpha=0.2)

plt.xlabel(r'$\gamma$', fontsize=16)
plt.ylabel('Average Quality Ratio', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xscale('log')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.axhline(y=1, lw=0.5, color='black')
plt.legend(fontsize=14, loc='upper center')
plt.tight_layout()
plt.savefig('fig_targeting.pdf')


# Above is main experiment
# 
# ---
# 
# Everything below is supplementary testing and analyses

# In[ ]:


# plot Fig3

xs = phis
ys = gammas
phis1 = phis
phis2 = phis
wires = gammas
new_wires = gammas
cmap = None
xlabel = '$\\phi$'
ylabel = '$\\gamma$'

kendall_pic_title = 'Discriminative power'
avg_quality_pic_title = 'Average Quality'
diversity_pic_title = 'Diversity'

figure = plt.figure(figsize=(13, 15), facecolor='w')
markers = ["o", "s", "^"]

save_dir = "results/prefer"

### 1. average quality ###
if save_dir == "results/random":
    file_template = "{}/avg_quality_random_phi{}_gamma{}.pkl"
else:
    file_template = "{}/avg_quality_prefer_phi{}_gamma{}.pkl"

# distr plot
ax = figure.add_subplot(3,2,1)
for idx, phi in enumerate(phis1):
    avg_qualities = []
    stds = []
    for gamma in wires:
        fname = file_template.format(save_dir, phi, gamma)
        fp = open(fname, "rb")
        data = pickle.load(fp)
        fp.close()
        avg_qualities.append(np.mean(data))
        stds.append(np.std(data))
    ax.plot(new_wires, avg_qualities, marker=markers[idx], label='$\\phi$:'+str(h))

ax.set_xlabel('$\\gamma$', fontsize=14)
ax.set_ylabel('Average quality', fontsize=14)
ax.set_xscale('log')
ax.set_xlim((new_wires[0], new_wires[-1]))
ax.set_xlim((new_wires[0], new_wires[-1]))
ax.legend(loc='upper right', fontsize=14)

# heatmap plot
ax = figure.add_subplot(3,2,2)
grid = np.zeros((len(wires), len(phis2)))
for i, gamma in enumerate(wires):
    for j, phi in enumerate(phis2):
        fname = file_template.format(save_dir, phi, gamma)
        fp = open(fname, "rb")
        data = pickle.load(fp)
        fp.close()
        grid[i, j] = np.mean(data)
draw_heatmap(ax, grid, xs, ys, xlabel, ylabel, cmap, avg_quality_pic_title, vmin=None, vmax=None)


### 2. average diversity ###
if save_dir == "results/random":
    file_template = "{}/avg_diversity_prefer_phi{}_gamma{}.pkl"
else:
    file_template = "{}/avg_diversity_prefer_phi{}_gamma{}.pkl"

# distr plot
ax = figure.add_subplot(3,2,3)
for idx, phi in enumerate(phis1):
    avg_diversities = []
    stds = []
    for gamma in wires:
        fname = file_template.format(save_dir, phi, gamma)
        fp = open(fname, "rb")
        data = pickle.load(fp)
        fp.close()
        avg_diversities.append(np.mean(data))
        stds.append(np.std(data))
    ax.plot(new_wires, avg_diversities, marker=markers[idx], label='$\\phi$:'+str(h))

ax.set_xlabel('$\\gamma$', fontsize=14)
ax.set_ylabel('Diversity', fontsize=14)
ax.set_xscale('log')
ax.set_xlim((new_wires[0], new_wires[-1]))
ax.set_xlim((new_wires[0], new_wires[-1]))

# heatmap plot
ax = figure.add_subplot(3,2,4)
grid = np.zeros((len(wires), len(phis2)))
for i, gamma in enumerate(wires):
    for j, phi in enumerate(phis2):
        fname = file_template.format(save_dir, phi, gamma)
        fp = open(fname, "rb")
        data = pickle.load(fp)
        fp.close()
        grid[i, j] = np.mean(data)
draw_heatmap(ax, grid, xs, ys, xlabel, ylabel, cmap, diversity_pic_title, vmin=None, vmax=None)

### 3. kendall ###
if save_dir == "results/random":
    file_template = "{}/kendall_random_phi{}_gamma{}.pkl"
else:
    file_template = "{}/kendall_prefer_phi{}_gamma{}.pkl"

# distr plot
ax = figure.add_subplot(3,2,5)
for idx, phi in enumerate(phis1):
    kendalls = []
    stds = []
    for gamma in wires:
        fname = file_template.format(save_dir, phi, gamma)
        fp = open(fname, "rb")
        data = pickle.load(fp)
        fp.close()
        kendalls.append(np.mean(data))
        stds.append(np.std(data))
    ax.plot(new_wires, kendalls, marker=markers[idx], label='$\\phi$:'+str(h))

ax.set_xlabel('$\\gamma$', fontsize=14)
ax.set_ylabel('Discriminative power', fontsize=14)
ax.set_xscale('log')
ax.set_xlim((new_wires[0], new_wires[-1]))
ax.set_xlim((new_wires[0], new_wires[-1]))
# ax.legend(loc='lower left', fontsize=14)

# heatmap plot
ax = figure.add_subplot(3,2,6)
grid = np.zeros((len(wires), len(phis2)))
for i, gamma in enumerate(wires):
    for j, phi in enumerate(phis2):
        fname = file_template.format(save_dir, phi, gamma)
        fp = open(fname, "rb")
        data = pickle.load(fp)
        fp.close()
        grid[i, j] = np.mean(data)
draw_heatmap(ax, grid, xs, ys, xlabel, ylabel, cmap, kendall_pic_title, vmin=None, vmax=None)

### 4. save plot ###
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
plt.savefig(save_dir + "/all_distr_heatmap.png")


# In[ ]:


# plot Fig5

save_dir = "results/random"
if save_dir == "results/random":
    file_template = "{}/tracked_memes_random_phi{}_gamma{}.pkl"
else:
    file_template = "{}/tracked_memes_prefer_phi{}_gamma{}.pkl"

fig, axs = plt.subplots(2, 3, figsize=(14, 8))
for i, phi in enumerate([1, 10]):
    for j, gamma in enumerate([0.001, 0.005, 0.01]):
        fname = file_template.format(save_dir, phi, gamma)
        fp = open(fname, "rb")
        data = pickle.load(fp)
        fp.close()

        quality, number_selected = zip(*data)

        low_quality_pop = []
        high_quality_pop = []
        for qua, pop in zip(quality, number_selected):
            if qua > 0:
                high_quality_pop.append(pop)
            else:
                low_quality_pop.append(pop)

        count = get_count(high_quality_pop)
        distr, sum_ = get_distr(count)
        h_mids, h_heights = getbins(distr, sum_)

        count = get_count(low_quality_pop)
        distr, sum_ = get_distr(count)
        l_mids, l_heights = getbins(distr, sum_)

        h_dict = defaultdict(list)
        for hm, hh in zip(h_mids, h_heights):
            h_dict[hm].append(hh)
        l_dict = defaultdict(list)
        for lm, lh in zip(l_mids, l_heights):
            l_dict[lm].append(lh)

        hs = []
        for k, v in h_dict.items():
            hs.append([k, np.mean(v)])
        h_mids, h_heights = zip(*sorted(hs, key=lambda x:x[0]))
        ls = []
        for k, v in l_dict.items():
            ls.append([k, np.mean(v)])
        l_mids, l_heights = zip(*sorted(ls, key=lambda x:x[0]))

        ax = axs[i][j]
        ax.loglog(h_mids, h_heights, marker='s', label='high quality')
        ax.loglog(l_mids, l_heights, marker='^', label='low quality')
        ax.set_xlabel('popularity', fontsize=14)
        ax.set_ylabel('P(popularity)', fontsize=14)
        ax.tick_params(labelsize=14)
        ax.annotate('$\\gamma={}$\n$\\phi={}$'.format(gamma, phi), xy=(0.05, 0.05), xycoords='axes fraction', fontsize=12)
        if i == 0 and j == 0:
            ax.legend(loc="upper right", fontsize=15)

plt.subplots_adjust(left=0.08, right=0.92, top=0.92, wspace=0.3, hspace=0.3)
plt.show()
plt.savefig(save_dir + "meme_quality_random_distr.png")
plt.close()


# In[ ]:


# plot Fig6

save_dir = "results/random"
if save_dir == "results/random":
    file_template = "{}/bad_memes_selected_time_random_phi{}_gamma{}.pkl"
else:
    file_template = "{}/bad_memes_selected_time_prefer_phi{}_gamma{}.pkl"

for i, phi in enumerate([1]):
    plt.figure(figsize=(10, 5))
    for j, gamma in enumerate([0.001]): #[0.5]
        fname = file_template.format(save_dir, phi, gamma)
        fp = open(fname, "rb")
        data = pickle.load(fp)
        fp.close()
        
        good_selected = []
        bad_selected = []
        for _, value in data.items():
            if value[0] <= 0 or value[1] <= 0:
                continue
            good_selected.append(value[0])
            bad_selected.append(value[1])

        count = dict([val for val in zip(bad_selected, good_selected)])
        distr_x, distr_y = get_distr(count)
        mids, heights = getbins(distr_x, distr_y)
        ratios = [np.log(height_)/np.log(mid_) for height_, mid_ in zip(heights, mids)]

        plt.subplot(121)
        plt.loglog(mids, heights, marker='o', label='$\\gamma$:'+str(gamma))
        plt.subplot(122)
        plt.plot(mids, ratios, marker='o', label='$\\gamma$:'+str(gamma))
        plt.xscale('log')
    
    # save fig
    plt.subplot(121)
    plt.loglog([min(mids), max(mids)], [min(mids), max(mids)], '--')
    plt.xlabel("Bot posts per meme", fontsize=14)
    plt.ylabel("Human posts per meme", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.margins(0.1)
    plt.legend(loc='best', fontsize=14)

    plt.subplot(122)
    plt.xlabel("Bot posts per meme", fontsize=14)
    plt.ylabel("Exponent $\\eta$", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.margins(0.1)
    plt.legend(loc='best', fontsize=14)

    plt.subplots_adjust(left=0.1, bottom=0.14, wspace=0.4)
    plt.show()
    plt.savefig(save_dir + "bad_meme_selected_random_distr_{}".format(phi))
    plt.close()


# In[ ]:




