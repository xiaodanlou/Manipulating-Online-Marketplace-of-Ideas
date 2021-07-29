#!/usr/bin/env python
# coding: utf-8

# Reproducing model in paper with Xiaodan: https://www.overleaf.com/project/5c375cf8f540a47e999968db
# 
# Notes:
# * Need Python 3.6 or later; eg: `module load python/3.6.6`
# * remember link direction is following, opposite of info spread!
# * using average mu instead of drawing from distribution; updated paper accordingly
# * measuring average quality just once at steady state; updated paper accordingly
# * Merge or replace Xiaodan's code (https://github.com/xiaodanlou/Information-Pollution-by-Social-Bots). Note this notebook does not include code to calculate diversity and kendall tau (Fig 3), the popularity distributions (Fig 5), or the amplification (Fig 6).
# 

# In[1]:


import networkx as nx
import random
import numpy
import math
import statistics
import csv
import matplotlib.pyplot as plt
from operator import itemgetter
import sys
import fcntl
import time

get_ipython().run_line_magic('matplotlib', 'inline')
assert(nx.__version__ >= '2.4')


# In[20]:


# parameters and utility globals

n_humans = 1000 # 10k for paper
beta = 0.1 # bots/humans ratio; 0.1 for paper
p = 0.5 # for network clustering; 0.5 for paper
k_out = 3 # average no. friends within humans & bots; 3 for paper
alpha = 15 # depth of feed; 15 for paper
mu = 0.75 # average prob of new meme vs retweet; 0.75 for paper or draw from empirical distribution
phi = 1 # bot deception >= 1: meme fitness higher than quality 
gamma = 0.1 # infiltration: probability that a human follows each bot
epsilon = 0.01 # threshold used to check for steady-state convergence
n_runs = 20 # number of simulations to average results
cvsfile = 'results.csv' # to save results for plotting

# if called with gamma as a command line params
if len(sys.argv) == 2:
  gamma = float(sys.argv[1])
  assert(0 <= gamma <= 1)


# In[21]:


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


# In[22]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


# count the number of forgotten memes as a function of in_degree (followers)
# using a global variable that needs to be reset

forgotten_memes = {}
def forgotten_memes_per_degree(n_forgotten, followers):
  if followers in forgotten_memes:
    forgotten_memes[followers] += n_forgotten
  else:
    forgotten_memes[followers] = n_forgotten


# In[8]:


# a single simulation step in which one agent is activated

def simulation_step(G, count_forgotten_memes=False):
  agent = random.choice(list(G.nodes()))
  memes_in_feed = G.nodes[agent]['feed']
  if len(memes_in_feed) and random.random() > mu:
    # retweet a meme from feed selected on basis of its fitness
    fitnesses = [m[1] for m in memes_in_feed]
    meme = random.choices(memes_in_feed, weights=fitnesses, k=1)[0]
  else:
    # new meme
    meme = get_meme(G.nodes[agent]['bot'])
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


# In[9]:


# calculate average quality of memes in system

def measure_average_quality(G):
  total = 0
  count = 0
  for agent in G.nodes:
    if G.nodes[agent]['bot'] == False:
      for m in G.nodes[agent]['feed']:
        count += 1
        total += m[0]
  return total / count


# In[10]:


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


# In[11]:


# new network from old but replace feed with average quality (used for Gephi viz)

def add_avq_to_net(G):
  newG = G.copy()
  for agent in newG.nodes:
    if len(newG.nodes[agent]['feed']) < 1:
      print('Bot' if newG.nodes[agent]['bot'] else 'Human', 'has empty feed')
    newG.nodes[agent]['feed'] = float(statistics.mean([m[0] for m in G.nodes[agent]['feed']]))
  return newG


# In[11]:


# main simulation 
# steady state is determined by small relative change in average quality
# returns average quality at steady state 

def simulation(preferential_targeting_flag, return_net=False, count_forgotten=False, network=None, verbose=False):
  if network is None:
    network = init_net(preferential_targeting_flag)
  n_agents = nx.number_of_nodes(network)
  old_quality = 100
  new_quality = 200
  time_steps = 0
  while max(old_quality, new_quality) > 0 and abs(new_quality - old_quality) / max(old_quality, new_quality) > epsilon: 
    if verbose:
      print('time_steps = ', time_steps, ', q = ', new_quality, flush=True) 
    time_steps += 1
    for _ in range(n_agents):
      simulation_step(network, count_forgotten_memes=count_forgotten)
    old_quality = new_quality
    new_quality = measure_average_quality(network)
  if return_net:
    return (new_quality, network)
  else:
    return new_quality


# In[12]:


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


# In[13]:


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


# Above are definitions
# 
# ---
# 
# Below is main experiment

# In[ ]:


# experiment, save results to CSV file
# this is slow for large n_humans; better to run in parallel 
# on a server or cluster, eg, one process per gamma value

for gamma in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
  q_random = []
  q_preferential = []
  q_ratio = []
  for sim in range(n_runs):
    print('Running Simulation ', sim, ' for gamma = ', gamma, ' ...', flush=True)
    qr = simulation(False)
    qp = simulation(True)
    q_random.append(qr)
    q_preferential.append(qp)
    q_ratio.append(qp/qr)

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

_, _, _, _, ratio_mu75, stderr_mu75 = read_csv('results_N1000.csv')
ymin_mu75 = [ratio_mu75[x] - 2*stderr_mu75[x] for x in ratio_mu75.keys()]
ymax_mu75 = [ratio_mu75[x] + 2*stderr_mu75[x] for x in ratio_mu75.keys()]
plt.plot(list(ratio_mu75.keys()), list(ratio_mu75.values()), label=r'$\mu=0.75$')
plt.fill_between(list(ratio_mu75.keys()), ymax_mu75, ymin_mu75, alpha=0.2)

_, _, _, _, ratio_mu25, stderr_mu25 = read_csv('results_N1000_mu0.25.csv')
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
# Below are supplementary testing and analyses

# In[ ]:


# TESTING

# print("random targeting: average quality = ", simulation(False))
# print("prefer targeting: average quality = ", simulation(True))

# gamma = 0.5
# G_random = init_net(False)
# G_prefer = init_net(True)
# print("random targeting: density = ", nx.density(G_random))
# print("prefer targeting: density = ", nx.density(G_prefer))

# export GML files for figures
gamma = 0.002
phi = 5
_, G_random_002_5 = simulation(False, return_net=True)
nx.write_gml(add_avq_to_net(G_random_002_5),'G_random_002_5.gml')
#_, G_prefer_002 = simulation(True, return_net=True)
#nx.write_gml(add_avq_to_net(G_prefer_002),'G_prefer_002.gml')
gamma = 0.02
phi = 1
_, G_random_02 = simulation(False, return_net=True)
nx.write_gml(add_avq_to_net(G_random_02),'G_random_02.gml')
phi = 5
_, G_random_02_5 = simulation(False, return_net=True)
nx.write_gml(add_avq_to_net(G_random_02_5),'G_random_02_5.gml')
#_, G_prefer_2 = simulation(True, return_net=True)
#nx.write_gml(add_avq_to_net(G_prefer_2),'G_prefer_2.gml')


# In[ ]:


# TEST NO BOTS

beta = 0.0
gamma = 0.0
av_q, G_no_bots = simulation(False, return_net=True)
nx.write_gml(add_avq_to_net(G_no_bots),'G_no_bots.gml')
print("Average quality without bots:", av_q)


# In[ ]:


# calculate Gini coefficient of concentration of low-quality memes around hubs
# inspired by https://github.com/oliviaguest/gini

def gini(G):
  humans = []
  total = 0
  for agent in G.nodes:
    if G.nodes[agent]['bot'] == False:
      zeros = 0
      for m in G.nodes[agent]['feed']:
        if m[0] == 0: 
          zeros += 1 
      humans.append((G.in_degree(agent), zeros))
      total += zeros
  humans.sort(key=itemgetter(0))
  n = len(humans)
  coefficient = 0
  for i in range(n):
    coefficient += (2*(i+1) - n - 1) * humans[i][1]
  return coefficient / (n * total)


# In[ ]:


# test Gini concentration of zeros

for gamma in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
  _, G = simulation(False, return_net=True)
  gini_coefficient_random = gini(G)
  print("random targeting, gamma =", gamma, ": Gini =", gini_coefficient_random)
  _, G = simulation(True, return_net=True)
  gini_coefficient_prefer = gini(G)
  print("prefer targeting, gamma =", gamma, ": Gini =", gini_coefficient_prefer)
  print("prefer targeting, gamma =", gamma, ": Gini diff=", gini_coefficient_prefer-gini_coefficient_random)


# In[ ]:


# STEPWISE TEST INIT

network = init_net(True) # only once!


# In[ ]:


# STEPWISE TEST STEP

simulation_step(network)
print("prefer targeting: average quality =", measure_average_quality(network))
for agent in network.nodes:
  if network.nodes[agent]['bot'] == False:
    print('human feed:', ["{0:.2f}".format(round(m[0], 2)) for m in network.nodes[agent]['feed']])


# In[ ]:


# relationship between indegree (#followers) and low-quality in humans

def quality_vs_degree(G):
  avg_quality = {}
  n_zeros = {}
  for agent in G.nodes:
    if G.nodes[agent]['bot'] == False:
      count = 0
      total = 0
      zeros = 0
      for m in G.nodes[agent]['feed']:
        count += 1
        total += m[0]
        if m[0] == 0:
          zeros += 1
      k = G.in_degree(agent)
      if count > 0:
        if k not in avg_quality:
          avg_quality[k] = []
          n_zeros[k] = []
        avg_quality[k].append(total/count)
        n_zeros[k].append(zeros)
  for k in avg_quality:
    avg_quality[k] = statistics.mean(avg_quality[k])
  for k in n_zeros:
    n_zeros[k] = statistics.mean(n_zeros[k])
  return(avg_quality, n_zeros)


# In[ ]:


# count number of humans who follow at least one bot

def bot_followers(G):
  n = 0
  for agent in G.nodes:
    if G.nodes[agent]['bot'] == False:
      for friend in G.successors(agent):
        if G.nodes[friend]['bot']:
          n += 1
          break
  return n


# In[ ]:


# plot some quantity in a dictionary where keys are no. followers 

def plot_quantity_vs_degree(title, ylabel, data_dict):
  plt.figure()
  plt.xlabel('Followers', fontsize=16)
  plt.ylabel(ylabel, fontsize=16)
  plt.title(title, fontsize=16)
  #plt.xscale('log')
  plt.plot(*zip(*sorted(data_dict.items())))


# In[ ]:


# exp for relationship between indegree and low-quality in humans

gamma = 0.1
(avg_q, G) = simulation(False, return_net=True) 
(avg_quality, n_zeros) = quality_vs_degree(G)
plot_quantity_vs_degree('Random Targeting', 'Average Quality', avg_quality)
plot_quantity_vs_degree('Random Targeting', 'Number of Zeros', n_zeros)
(avg_q, G) = simulation(True, return_net=True)
(avg_quality, n_zeros) = quality_vs_degree(G)
plot_quantity_vs_degree('Preferential Targeting', 'Average Quality', avg_quality)
plot_quantity_vs_degree('Preferential Targeting', 'Number of Zeros', n_zeros)


# In[ ]:


# exp to see how the number of bot followers depends on gamma 

victims_ratio = {}
for gamma in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
  net_pref = init_net(True)
  net_rand = init_net(False)
  victims_ratio[gamma] = bot_followers(net_pref) / bot_followers(net_rand)
plt.xlabel(r'$\gamma$', fontsize=16)
plt.ylabel('Bot Followers Ratio', fontsize=16)
plt.xscale('log')
plt.plot(list(victims_ratio.keys()), list(victims_ratio.values()))


# In[ ]:


# exp for forgotten zero-quality memes in humans
# and relationship with indegree  

gamma = 0.5
n_zeros_forgotten_random = []
n_zeros_forgotten_prefer = []

for _ in range(n_runs):
  forgotten_memes = {}
  simulation(False, count_forgotten=True) 
  #print('Random Targeting:', sum(forgotten_memes.values()), 'low-quality memes forgotten')
  n_zeros_forgotten_random.append(sum(forgotten_memes.values()))

  forgotten_memes = {}
  simulation(True, count_forgotten=True)
  #print('Preferential Targeting:', sum(forgotten_memes.values()), 'low-quality memes forgotten')
  n_zeros_forgotten_prefer.append(sum(forgotten_memes.values()))

print('Random Targeting:', statistics.mean(n_zeros_forgotten_random), '+/-', statistics.stdev(n_zeros_forgotten_random) / math.sqrt(n_runs), 'low-quality memes forgotten')
print('Preferential Targeting:', statistics.mean(n_zeros_forgotten_prefer), '+/-', statistics.stdev(n_zeros_forgotten_prefer) / math.sqrt(n_runs), 'low-quality memes forgotten')


# In[ ]:


# similar to main simulation but run for fixed time and return avg quality over time

max_time_steps = 10

def simulation_timeline(preferential_targeting_flag):
  quality_timeline = []
  for time_steps in range(max_time_steps):
     quality_timeline.append([])
  for _ in range(n_runs):
    network = init_net(preferential_targeting_flag)
    n_agents = nx.number_of_nodes(network)
    for time_steps in range(max_time_steps):
      for _ in range(n_agents):
        simulation_step(network, count_forgotten_memes=False)
      quality = measure_average_quality(network)
      quality_timeline[time_steps].append(quality)
  for time_steps in range(max_time_steps):
    quality_timeline[time_steps] = statistics.mean(quality_timeline[time_steps])
  return quality_timeline 


# In[ ]:


# experiment plotting quality over time

for gamma in [0.01, 0.02, 0.03, 0.04, 0.05]: 
  print('gamma =', gamma)
  plt.figure()
  plt.xlabel('time steps', fontsize=16)
  plt.ylabel('average quality', fontsize=16)
  plt.title('gamma = ' + str(gamma), fontsize=16)
  timeline_random = simulation_timeline(False)
  timeline_prefer = simulation_timeline(True)
  plt.plot(timeline_random, label='Random Targeting')
  plt.plot(timeline_prefer, label='Preferential Targeting')
  plt.legend()


# In[ ]:


# DRAW NETWORK (better use Gephi)

G = init_net(False)
s = [G.in_degree(n) for n in G]
c = ['red' if G.nodes[n]['bot'] else 'blue' for n in G]
nx.draw(G, node_color = c, node_size = s)


# # Empirical Retweet Network

# In[14]:


# read retweet network from data and assign bot flags to nodes
# if we have bot score for a node from 'M5_centralities.csv', use it
# else if we have the bot score from 'user_bot_score.calibrated.csv', use that
# else assume node is human

# NB: we cannot add an empty list as a node attribute, so the feed will be added when reading the file

bot_score_raw = {}
with open('M5_centralities.csv') as file:
  next(file) # skip header line
  reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
  for row in reader:
    bot_score_raw[row[0]] = row[2]

bot_score_cal = {}
with open('user_bot_score.calibrated.csv') as file:
  next(file) # skip header line
  reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
  for row in reader:
    bot_score_cal[row[0]] = row[1]

retweets = nx.DiGraph()
with open('retweet.preelection.all.csv') as file:
  reader = csv.reader(file)
  next(reader) # skip header line
  for row in reader:
    n1 = int(row[0])
    n2 = int(row[1])
    retweets.add_edge(n1, n2)
    if n1 in bot_score_raw:
      bot_score = bot_score_raw[n1]
    elif n1 in bot_score_cal:
      bot_score = bot_score_cal[n1]
    else:
      bot_score = 0 # assuming human by default
    if bot_score > 0.5:
      retweets.nodes[n1]['bot'] = True
    else:
      retweets.nodes[n1]['bot'] = False
    if n2 in bot_score_raw:
      bot_score = bot_score_raw[n2]
    elif n2 in bot_score_cal:
      bot_score = bot_score_cal[n2]
    else:
      bot_score = 0 # assuming human by default
    if bot_score > 0.5:
      retweets.nodes[n2]['bot'] = True
    else:
      retweets.nodes[n2]['bot'] = False

print('the retweet network has', retweets.number_of_nodes(), 'nodes')
# 346,573


# In[15]:


# SAVE EMPIRICAL RETWEET NETWORK TO FILE for faster reading in subsequent experiments

retweet_network_file = 'retweet_network.gml'

nx.write_gml(retweets, retweet_network_file)


# In[ ]:


# RUN SIMULATION TO CALCULATE AVG QUALITY

avg_quality = simulation(False, network=retweets, verbose=True)
print('average quality for empirical network:', avg_quality)
# 0.3310


# In[5]:


# READ EMPIRICAL NETWORK FROM FILE

def read_retweet_network(file, add_feed=True):
    retweets = nx.read_gml(file)
    if add_feed:
        for n in retweets.nodes:
            retweets.nodes[n]['feed'] = []
    return retweets


# In[22]:


# CALCULATE beta AND gamma
# returns (n_bots, n_humans, beta, gamma)

def calculate_beta_gamma(RT):
    n_bots = 0
    for n in RT.nodes:
        if RT.nodes[n]['bot']:
            n_bots += 1
    n_humans = RT.number_of_nodes() - n_bots
    sum_of_of_gammas = 0
    for n in RT.nodes:
        if not RT.nodes[n]['bot']:
            bot_friends = 0
            for friend in RT.successors(n):
                if RT.nodes[friend]['bot']:
                    bot_friends += 1
            sum_of_of_gammas += bot_friends / n_bots    
    return(n_bots, n_humans, n_bots / n_humans, sum_of_of_gammas / n_humans)


# In[24]:


RT = read_retweet_network('retweet_network.gml')
n_bots, n_humans, beta, gamma = calculate_beta_gamma(RT)
print (n_bots, 'bots and', n_humans, 'humans, for a ratio of beta =', beta)
print ('humans on average follow a fraction gamma =', gamma, 'of the bots')

# 3351 bots and 343222 humans, for a ratio of beta = 0.009763360157565657
# humans on average follow a fraction gamma = 1.9402037768854127e-05 of the bots


# In[30]:


# sample to change the ratio of bots to humans (beta)

# initial network from file: extract bots and humans
RT = read_retweet_network('retweet_network.gml', add_feed=False)
n_bots, n_humans, beta, gamma = calculate_beta_gamma(RT)
bots = set([n for n, attr in RT.nodes(data=True) if attr['bot']])
humans = set([n for n, attr in RT.nodes(data=True) if not attr['bot']])
print('bots =', n_bots, 'humans =', n_humans, 'beta =', beta, 'gamma =', gamma)
filename = 'RT_h_' + str(n_humans) + '_b_' + str(n_bots) + '.gml'
nx.write_gml(RT, filename)

# downsample bots to decrease beta
RT_SAMPLE = RT.copy() # NOTE: the feed attributes will point to and be shared with the original!
bots_remaining = n_bots
while bots_remaining > 100:
    bots_to_delete = random.sample(bots, int(bots_remaining/2))
    bots = bots.difference(bots_to_delete)
    RT_SAMPLE.remove_nodes_from(bots_to_delete) 
    bots_remaining, humans_remaining, beta, gamma = calculate_beta_gamma(RT_SAMPLE)
    print('bots =', bots_remaining, 'humans =', humans_remaining, 'beta =', beta, 'gamma =', gamma)
    filename = 'RT_h_' + str(humans_remaining) + '_b_' + str(bots_remaining) + '.gml'
    nx.write_gml(RT_SAMPLE, filename)
    
# downsample humans to increase beta
RT_SAMPLE = RT.copy() # NOTE: the feed attributes will point to and be shared with the original!
humans_remaining = n_humans
while humans_remaining > 10000:
    humans_to_delete = random.sample(humans, int(humans_remaining/2))
    humans = humans.difference(humans_to_delete)
    RT_SAMPLE.remove_nodes_from(humans_to_delete) 
    bots_remaining, humans_remaining, beta, gamma = calculate_beta_gamma(RT_SAMPLE)
    print('bots =', bots_remaining, 'humans =', humans_remaining, 'beta =', beta, 'gamma =', gamma)
    filename = 'RT_h_' + str(humans_remaining) + '_b_' + str(bots_remaining) + '.gml'
    nx.write_gml(RT_SAMPLE, filename)


# In[ ]:


# MAIN: RUN SIMULATION TO CALCULATE AVG QUALITY using input network
# see empirical_beta.job and empirical_beta.py to run these in parallel on carbonate

for netfile in ['RT_h_10726_b_3351.gml','RT_h_343222_b_210.gml','RT_h_42903_b_3351.gml',
                'RT_h_171611_b_3351.gml','RT_h_343222_b_3351.gml','RT_h_5363_b_3351.gml',
                'RT_h_21452_b_3351.gml','RT_h_343222_b_419.gml','RT_h_85806_b_3351.gml',
                'RT_h_343222_b_105.gml','RT_h_343222_b_53.gml','RT_h_343222_b_1676.gml',
                'RT_h_343222_b_838.gml']:
    sampled_net = read_retweet_network(netfile)
    n_bots, n_humans, beta, gamma = calculate_beta_gamma(sampled_net)
    avg_quality = simulation(False, network=sampled_net)
    save_csv([n_bots, n_humans, beta, gamma, avg_quality])


# In[8]:


# PLOT AVG_Q vs BETA (after running simulations on carbonate)

beta2q = {}
avg_q_values = []
with open('beta.csv', newline='') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        beta2q[row[2]] = row[4]

plt.xlabel(r'$\beta$', fontsize=16)
plt.ylabel('Average Quality', fontsize=16)
plt.xscale('log')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim((10**-4,1))
plt.ylim((0,0.4))
plt.plot(*zip(*sorted(beta2q.items())))


# In[ ]:


# replace bots by an equal number (same BETA) of synthetic bots 
# with GAMMA as a parameter using empirical bot attachment for humans
# NB: THIS IS SLOW; BETTER RUN IN PARALLEL ON CARBONATE

# initial network from file
RT = read_retweet_network('retweet_network.gml', add_feed=False)

# for each human calculate vulnerability = number of bot friends
# (use later for probabilities of following synthetic bots)
# and also calculate the average out-degree of the bots
avg_bot_kout = 0
n_bots = 0
n_humans = 0
bots = []
for n in RT.nodes:
    bot_friends = 0
    for friend in RT.successors(n):
        if RT.nodes[friend]['bot']:
            bot_friends += 1
    if RT.nodes[n]['bot']:
        n_bots += 1
        avg_bot_kout += bot_friends
        bots.append(n)
    else:
        n_humans += 1
        RT.nodes[n]['vulnerability'] = bot_friends
avg_bot_kout /= n_bots
#print('avg_bot_kout =', avg_bot_kout) # avg_bot_kout = 3.245

#remove bots
RT.remove_nodes_from(bots)

# add synthetic bots; we preserve the empirical bot k_out, to the nearest integer 
# (happens to be 3 like in artificial networks :)
k_out = round(avg_bot_kout) 
B = random_walk_network(n_bots)
for b in B.nodes:
    B.nodes[b]['bot'] = True

# create new networks with synthetic bots and have vulnerable humans follow bots according to GAMMA
for gamma in [0.00001, 0.0001, 0.001, 0.01, 0.1]: 
    SYN = nx.disjoint_union(RT, B)
    humans = []
    bots = []
    weights = []
    for n in SYN.nodes:
        if SYN.nodes[n]['bot']:
            bots.append(n)
        else:
            humans.append(n)
            weights.append(SYN.nodes[n]['vulnerability'])
            del SYN.nodes[n]['vulnerability'] # no longer needed
    n_followers = round(n_humans * gamma)
    for b in bots:
        followers = sample_with_prob_without_replacement(humans, n_followers, weights)
        for f in followers:
            SYN.add_edge(f, b)
    filename = 'RT_gamma_' + str(gamma) + '.gml'
    nx.write_gml(SYN, filename)
    print('saved', filename)


# In[61]:


# PLOT AVG_Q vs GAMMA (after running simulations on carbonate)

gamma2q = {}
avg_q_values = []
with open('gamma.csv', newline='') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        gamma2q[row[3]] = row[4]

plt.xlabel(r'$\gamma$', fontsize=16)
plt.ylabel('Average Quality', fontsize=16)
plt.xscale('log')
plt.ylim((0,0.4))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(*zip(*sorted(gamma2q.items())))


# In[10]:


# PLOT AVG_Q vs PHI (after running experiments on carbonate)

phi2q = {}
avg_q_values = []
with open('phi.csv', newline='') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        phi2q[row[0]] = row[1]

plt.xlabel(r'$\phi$', fontsize=16)
plt.ylabel('Average Quality', fontsize=16)
plt.xlim((0.5,10.5))
plt.ylim((0,0.4))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(*zip(*sorted(phi2q.items())))


# In[62]:


## Combine the three plots

fig, (ax_beta, ax_gamma, ax_phi) = plt.subplots(1,3, sharey=True, figsize=(10,5))
for ax in fig.get_axes():
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylim((-0.01,0.36))

ax_beta.set_xlabel(r'$\beta$', fontsize=16)
ax_beta.set_xscale('log')
ax_beta.set_xlim((10**-4,1))
ax_beta.plot(*zip(*sorted(beta2q.items())))
ax_beta.set_ylabel('Average Quality', fontsize=16)

ax_gamma.set_xlabel(r'$\gamma$', fontsize=16)
ax_gamma.set_xscale('log')
ax_gamma.set_xlim((5*10**-6,0.2))
ax_gamma.set_xticks([10**-5, 10**-3, 0.1])
ax_gamma.plot(*zip(*sorted(gamma2q.items())))

ax_phi.set_xlabel(r'$\phi$', fontsize=16)
ax_phi.set_xlim((0.5,10.5))
ax_phi.set_xticks([1,4,7,10])
ax_phi.plot(*zip(*sorted(phi2q.items())))


# In[11]:


# PLOT AVG_Q vs FLOOD (after running experiments on carbonate)
# FLOOD (theta) is a multiplier for content posted by bots 

flood2q = {}
avg_q_values = []
with open('flood.csv', newline='') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        flood2q[row[0]] = row[1]

plt.xlabel(r'$\theta$', fontsize=16)
plt.ylabel('Average Quality', fontsize=16)
plt.xscale('log')
plt.ylim((0,0.4))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(*zip(*sorted(flood2q.items())))


# In[ ]:


# CHECK CORRELATION BETWEEN INFLUENCE (FOLLOWERS) AND VULNERABILITY (BOT FRIENDS) OF HUMANS IN EMPIRICAL NETWORK

RT = read_retweet_network('retweet_network.gml', add_feed=False)
influence = []
vulnerability = []
for n in RT.nodes:
    if not RT.nodes[n]['bot']:
        bot_friends = 0
        for friend in RT.successors(n):
            if RT.nodes[friend]['bot']:
                bot_friends += 1
        vulnerability.append(bot_friends)
        influence.append(RT.in_degree(n))


# In[13]:


from scipy.stats import pearsonr, spearmanr
r,pr = pearsonr(influence, vulnerability)
rho,prho = spearmanr(influence, vulnerability)
print('Pearson correlation =', r, '( p =', pr, ')')
print('Spearman correlation =', rho, '( p =', prho, ')')
# Pearson correlation = 0.0754934397277 ( p = 0.0 )
# Spearman correlation = 0.120199727675 ( p = 0.0 )

plt.xlabel('Followers', fontsize=16)
plt.ylabel('Bot Friends', fontsize=16)
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.scatter([x+1 for x in influence], [y+1 for y in vulnerability])


# In[ ]:




