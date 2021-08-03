#!/usr/bin/env python

##################################################################
# Code to (re)produce results in the paper 
# "Manipulating the Online Marketplace of Ideas" 
# by Xiaodan Lou, Alessandro Flammini, and Filippo Menczer
# https://arxiv.org/abs/1907.06130
# 
# Notes:
# * Need Python 3.6 or later; eg: `module load python/3.6.6`
# * remember link direction is following, opposite of info spread!
##################################################################

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
import bot_model

# create a network with random-walk growth model
# default p = 0.5 for network clustering
# default k_out = 3 is average no. friends within humans & bots
#
def random_walk_network(net_size, p=0.5, k_out=3):
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


# sample a bunch of objects from a list without replacement 
# and with given weights (to be used as probabilities), which can be zero
# NB: cannot use random_choices, which samples with replacement
#     nor numpy.random.choice, which can only use non-zero probabilities
#
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


# create network of humans and bots
# preferential_targeting is a flag; if False, random targeting
# default n_humans=1000 but 10k for paper
# default beta=0.1 is bots/humans ratio
# default gamma=0.1 is infiltration: probability that a human follows each bot
#
def init_net(preferential_targeting, n_humans=1000, beta=0.1, gamma=0.1):

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


# return (quality, fitness, id) meme tuple depending on bot flag
# using https://en.wikipedia.org/wiki/Inverse_transform_sampling
# default phi = 1 is bot deception; >= 1: meme fitness higher than quality 
# N.B. get_meme.id is an attribute that works as a static var to get unique IDs
#
def get_meme(bot_flag, phi=1):
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
  if hasattr(get_meme, 'id'):
    get_meme.id += 1
  else:
    get_meme.id = 0
  return (quality, fitness, get_meme.id)


# count the number of forgotten memes as a function of in_degree (followers)
# using dict attribute 'forgotten_memes' as a static variable
# that can be accessed as: forgotten_memes_per_degree.forgotten_memes
#
def forgotten_memes_per_degree(n_forgotten, followers):
  if not hasattr(forgotten_memes_per_degree, 'forgotten_memes'):
    forgotten_memes_per_degree.forgotten_memes = {} # initialize
  if followers in forgotten_memes_per_degree.forgotten_memes:
    forgotten_memes_per_degree.forgotten_memes[followers] += n_forgotten
  else:
    forgotten_memes_per_degree.forgotten_memes[followers] = n_forgotten


# track number of tweets and retweets of each meme
# using dict attribute 'popularity' as a static variable 
# that can be accessed as: track_memes.popularity and
# has prototype {(meme_tuple): popularity}
#
# in addition if quality == 0 we also track the popularity
# by bots and humans separately using another dict attribute 
# track_memes.bad_popularity as a static variable 
# with prototype {"meme_id": [human_popularity, bot_popularity]}
#
def track_memes(meme, bot_flag):
  if not hasattr(track_memes, 'popularity'):
    track_memes.popularity = {}
  if meme in track_memes.popularity:
    track_memes.popularity[meme] += 1
  else:
    track_memes.popularity[meme] = 1
  if meme[0] == 0:
    if not hasattr(track_memes, 'bad_popularity'):
      track_memes.bad_popularity = {}
    oneifbot = 1 if bot_flag else 0
    if meme[2] in track_memes.bad_popularity:
      track_memes.bad_popularity[meme[2]][oneifbot] += 1
    else:
      track_memes.bad_popularity[meme[2]] = [0,0]
      track_memes.bad_popularity[meme[2]][oneifbot] = 1


# a single simulation step in which one agent is activated
# default alpha = 15 is depth of feed
# default mu = 0.75 is average prob of new meme vs retweet; 
#         mu could also be drawn from empirical distribution
#
def simulation_step(G,
                    count_forgotten_memes=False,
                    track_meme=False,
                    alpha=15,
                    mu=0.75,
                    phi=1):

  agent = random.choice(list(G.nodes()))
  memes_in_feed = G.nodes[agent]['feed']
  
  # tweet or retweet
  if len(memes_in_feed) and random.random() > mu:
    # retweet a meme from feed selected on basis of its fitness
    fitnesses = [m[1] for m in memes_in_feed]
    meme = random.choices(memes_in_feed, weights=fitnesses, k=1)[0]
  else:
    # new meme
    meme = get_meme(G.nodes[agent]['bot'], phi)
  
  # bookkeeping
  if track_meme:
    track_memes(meme, G.nodes[agent]['bot'])

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


# calculate average quality of memes in system
#
def measure_average_quality(G, count_bot=False):
  total = 0
  count = 0
  for agent in G.nodes:
    if count_bot == True or G.nodes[agent]['bot'] == False:
      for m in G.nodes[agent]['feed']:
        count += 1
        total += m[0]
  return total / count


# calculate fraction of low-quality memes in system
#
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


# new network from old but replace feed with average quality
# (used for Gephi viz)
#
def add_avq_to_net(G):
  newG = G.copy()
  for agent in newG.nodes:
    if len(newG.nodes[agent]['feed']) < 1:
      print('Bot' if newG.nodes[agent]['bot'] else 'Human', 'has empty feed')
    newG.nodes[agent]['feed'] = float(statistics.mean([m[0] for m in G.nodes[agent]['feed']]))
  return newG


# main simulation 
# steady state is determined by small relative change in average quality
# returns average quality at steady state 
# default epsilon=0.01 is threshold used to check for steady-state convergence
#
def simulation(preferential_targeting_flag, 
               return_net=False,
               count_forgotten=False,
               track_meme=False,
               network=None, 
               verbose=False,
               epsilon=0.01,
               mu=0.75,
               phi=1,
               gamma=0.1):
  
  if network is None:
    network = init_net(preferential_targeting_flag, gamma=gamma)
  n_agents = nx.number_of_nodes(network)
  old_quality = 100
  new_quality = 200
  time_steps = 0
  while max(old_quality, new_quality) > 0 and abs(new_quality - old_quality) / max(old_quality, new_quality) > epsilon: 
    if verbose:
      print('time_steps = ', time_steps, ', q = ', new_quality, flush=True) 
    time_steps += 1
    for _ in range(n_agents):
      simulation_step(network,
                      count_forgotten_memes=count_forgotten,
                      track_meme=track_meme,
                      mu=mu, phi=phi) 
    old_quality = new_quality
    new_quality = measure_average_quality(network)
  if return_net:
    return (new_quality, network)
  else:
    return new_quality


# append to file, locking file and waiting if busy in case of multi-processing
#
def save_csv(data_array, csvfile='results.csv'): 
  with open(csvfile, 'a', newline='') as file:
    while True:
      try:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        writer = csv.writer(file)
        writer.writerow(data_array)
        fcntl.flock(file, fcntl.LOCK_UN)
        break
      except:
        time.sleep(0.1)


# read from file
#
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


# calculate log with default base
#
def logbase(x, base=1.5):
    return np.log(x)/np.log(base)


# histogram
# 
def get_count(list):
    count = {}
    for q in list:
        if q in count:
            count[q] += 1
        else:
            count[q] = 1
    return count


# log-bin given histogram
#
def get_distr(count):
    distr = {}
    sum = 0
    for a in count:
        sum += count[a]
        bin = int(logbase(a))
        if bin in distr:
            distr[bin] += count[a]
        else:
            distr[bin] = count[a]
    return distr, sum


# log-binned distribution given log-binned histogram with default base
#
def getbins(distr, sum, base=1.5):
    mids = []
    heights = []
    bin = sorted(distr.keys())
    for i in bin:
        start = base ** i
        width = base ** (i+1) - start
        mid = start + width/2
        mids.append(mid)
        heights.append(distr[i]/(sum * width))
    return mids, heights


# plot heatmap
#
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
    ax.set_xticklabels(xticks, fontsize=14) #, rotation=40
    cb = plt.colorbar(mappable=map, cax=None, ax=None)
    cb.ax.tick_params(labelsize=12)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title)


# calculate Gini coefficient of concentration of low-quality memes around hubs
# inspired by https://github.com/oliviaguest/gini
#
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


# relationship between indegree (#followers) and low-quality in humans
#
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


# count number of humans who follow at least one bot
#
def bot_followers(G):
  n = 0
  for agent in G.nodes:
    if G.nodes[agent]['bot'] == False:
      for friend in G.successors(agent):
        if G.nodes[friend]['bot']:
          n += 1
          break
  return n


# similar to main simulation but run for fixed time 
# and return avg quality over time
#
def simulation_timeline(preferential_targeting_flag, max_time_steps=10, gamma=0.1):
  quality_timeline = []
  for time_steps in range(max_time_steps):
     quality_timeline.append([])
  for _ in range(n_runs):
    network = init_net(preferential_targeting_flag, gamma=gamma)
    n_agents = nx.number_of_nodes(network)
    for time_steps in range(max_time_steps):
      for _ in range(n_agents):
        simulation_step(network, count_forgotten_memes=False)
      quality = measure_average_quality(network)
      quality_timeline[time_steps].append(quality)
  for time_steps in range(max_time_steps):
    quality_timeline[time_steps] = statistics.mean(quality_timeline[time_steps])
  return quality_timeline 


# plot some quantity in a dictionary where keys are no. followers 
#
def plot_quantity_vs_degree(title, ylabel, data_dict):
  plt.figure()
  plt.xlabel('Followers', fontsize=16)
  plt.ylabel(ylabel, fontsize=16)
  plt.title(title, fontsize=16)
  #plt.xscale('log')
  plt.plot(*zip(*sorted(data_dict.items())))


# READ EMPIRICAL NETWORK FROM GML FILE
#
def read_retweet_network(file, add_feed=True):
    retweets = nx.read_gml(file)
    if add_feed:
        for n in retweets.nodes:
            retweets.nodes[n]['feed'] = []
    return retweets


# CALCULATE beta AND gamma 
# returns (n_bots, n_humans, beta, gamma)
#
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

