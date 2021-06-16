#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))



def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3


# In[2]:



import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
from IPython.display import clear_output


if "../" not in sys.path:
    sys.path.append("../") 

from collections import defaultdict

matplotlib.style.use('ggplot')


# In[3]:


from tqdm import tqdm


# In[4]:


from Environment import *


# In[5]:


import pandas as pd
import random


class Qlearning:
    _qmatrix = None
    _learn_rate = None
    _discount_factor = None

    def __init__(self,
                 possible_states,
                 possible_actions,
                 initial_reward,
                 learning_rate,
                 discount_factor,
                epsilon):
        """
        Initialise the q learning class with an initial matrix and the parameters for learning.

        :param possible_states: list of states the agent can be in
        :param possible_actions: list of actions the agent can perform
        :param initial_reward: the initial Q-values to be used in the matrix
        :param learning_rate: the learning rate used for Q-learning
        :param discount_factor: the discount factor used for Q-learning
        """
        # Initialize the matrix with Q-values
        init_data = [[float(initial_reward) for _ in possible_states]
                     for _ in possible_actions]
        self._qmatrix = pd.DataFrame(data=init_data,
                                     index=possible_actions,
                                     columns=possible_states)
        self._qmatrix["count"] = 0

        # Save the parameters
        self._learn_rate = learning_rate
        self._discount_factor = discount_factor
        self.initial_reward = initial_reward
        self.epsilon = epsilon

    def get_best_action(self, state, actions,policy=True):
        """
        Retrieve the action resulting in the highest Q-value for a given state.

        :param state: the state for which to determine the best action
        :return: the best action from the given state
        """
        
        for x in actions:
            tag = x.split("!@!")[0]
            qv = len(actions)-len([y for y in actions if y.split("!@!")[0]==tag])
            qv = self.initial_reward+(self.initial_reward*qv)
            #self.checkAction(x,Qvalue=qv)
            self.checkAction(x)
        
        self.checkState(state,availableActions=actions)
        
        if policy:
            if random.random()>self.epsilon:
                # Return the action (index) with maximum Q-value
                return self._qmatrix[[state]].idxmax().iloc[0]
            else:
                ac = random.choice(self._qmatrix.index)
                return ac
        else:
            return self._qmatrix.loc[actions][[state]].idxmax().iloc[0]
    
    def checkState(self,state,Qvalue=-999999,availableActions=None):
        if Qvalue==-999999:
            Qvalue=self.initial_reward
        if state not in self._qmatrix.columns:
            #print("Adding State",state)
            if availableActions!=None:
                if type(availableActions)==list:
                    self._qmatrix[state] = -999999
                    self._qmatrix.loc[availableActions,state] = float(Qvalue)
            else:
                self._qmatrix[state] = float(Qvalue)
            
    def checkAction(self,action,Qvalue=-999999):
        if Qvalue==-999999:
            Qvalue=self.initial_reward
        if action not in self._qmatrix.index:
            #print("Adding Action",action)
            self._qmatrix.loc[action] = float(Qvalue)
            self._qmatrix.loc[action,"count"] = 0

    def update_model(self, state, action, reward, next_state,next_actions):
        """
        Update the Q-values for a given observation.

        :param state: The state the observation started in
        :param action: The action taken from that state
        :param reward: The reward retrieved from taking action from state
        :param next_state: The resulting next state of taking action from state
        """
        self.checkAction(action)
        self.checkState(state)
        
        # Update q_value for a state-action pair Q(s,a):
        # Q(s,a) = Q(s,a) + α( r + γmaxa' Q(s',a') - Q(s,a) )
        #print("Updating Temporal Difference")
        q_sa = self._qmatrix.loc[action, state]
        if len(next_actions)>=1:
            max_q_sa_next = self._qmatrix.loc[self.get_best_action(next_state,next_actions,policy=False), next_state]
            r = reward
            alpha = self._learn_rate
            gamma = self._discount_factor*np.exp(-0.1*(len(next_actions)-1))
            # Do the computation
            new_q_sa = q_sa + alpha * (r + gamma * max_q_sa_next - q_sa)
            if type(new_q_sa)==pd.core.series.Series:
                new_q_sa = new_q_sa.values[0]
            #print("newq=",new_q_sa)
            self._qmatrix.loc[action, state] = new_q_sa
            return 1
        #print("updated")
        else:
            max_q_sa_next = -999999
            r = reward
            alpha = self._learn_rate
            gamma = self._discount_factor*np.exp(-0.1*(0-1))
            # Do the computation
            new_q_sa = q_sa + alpha * (r + gamma * max_q_sa_next - q_sa)
            if type(new_q_sa)==pd.core.series.Series:
                new_q_sa = new_q_sa.values[0]
            #print("newq=",new_q_sa)
            self._qmatrix.loc[action, state] = new_q_sa
            return -1
        
        
        
from apted import APTED
from apted.helpers import Tree
import pandas as pd
import numpy as np
import glob, os
from tqdm import tqdm
import itertools
from bs4 import BeautifulSoup
import bs4
#from collections import defaultdict

from collections import OrderedDict, Callable

class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))



def defaultVal():
    return [[],0]

def makeTree(S):
    S = S.strip()
    S = S.replace("\n","")
    #S = S.replace(" ","")
    S = S.replace("\t","")
    S = S.replace("\r","")
    soup = BeautifulSoup(S, "html.parser")
    return soup

def recursiveChildBfs(bs):
    root = bs
    stack = [root]
    count=0
    parrent = [None]
    while len(stack) != 0:
        node = stack.pop(0)
        pnode = parrent.pop(0)
        if node is not bs:
            if node.name!=None:
                yield node.name+"~"+str(count),pnode
            else:
                yield node.name,pnode
        if hasattr(node, 'children'):
            for child in node.children:
                stack.append(child)
                parrent.append(node.name+"~"+str(count))
        count+=1

def visit(tagdict,c,tree):
    tree+="{"
    tree+=c.split("~")[0]
    for i in tagdict[c][0]:
        tree = visit(tagdict,i,tree)
        tree+="}"
    return tree        

def generateTree(S):
    html = makeTree(S)
    tagdict = DefaultOrderedDict(defaultVal)
    for c,p in recursiveChildBfs(html):
        if c!=None:
            tagdict[p][0].append(c)
            tagdict[p][1]+=1


    tree = "{"
    for x,y in zip(list(tagdict.keys())[1::],list(tagdict.values())[1::]):
        tree+=x.split("~")[0]
        for c in y[0]:
            #tree+="{"
            #tree+=c
            tree = visit(tagdict,c,tree)
            tree+="}"
        tree+="}"
        break
    nNodes = 0
    for x in tagdict.keys():
        nNodes+=tagdict[x][1]
    return tree,nNodes


# In[6]:


import time
import numpy as np
import pandas as pd
import os
import shutil
from collections import defaultdict
import pickle
import json
from threading import Thread

CLOSE=False

def save_stateMap(obj, name):
    with open(name,"w") as dd:
        dd.write(json.dumps(obj))

def load_stateMap(name):
    with open(name) as json_file:
        data = defaultdict(factory)
        data2 = json.load(json_file)
        data.update(data2)
    return data

def factory():
    return {"src":"","edges":[],"url":"","start":0}

def makeGraph(stateMap,output):
    statesfile = [file.split("/")[-1].split(".html")[0] for file in glob.glob(output+"/*.html")]
    with open(os.path.join(output,"data.js"),"w") as jsonwriter:
        C = []
        for x in stateMap.keys():
            stateMap[x]["edges"] = [{"action":y["action"],"state":y["state"]} for y in stateMap[x]["edges"] if y["state"] in statesfile]
            C.append(stateMap[x])
        C.sort(key=lambda x:x["start"],reverse=True)
        jsonwriter.write("let data = ")
        jsonwriter.write(json.dumps(C))

def q_learning(env, num_episodes,sleep=0,matrix=None,statemap=None,_epsilon=0.2,
               onlyperform=False,output="./Q_Result",timebound=True,activity_time=None,
              login_urls=[],username="",password="",depth=100):
    global CLOSE
    #stats = EpisodeStats(
    #    episode_lengths=np.zeros(num_episodes),
    #    episode_rewards=np.zeros(num_episodes))    
    
    env.reset()
    state = env.get_Actions_OR_state()
    state = chash.md5(state)
    
    Qlearn = Qlearning(possible_states = [state],
                      possible_actions = env.get_Actions_OR_state(True),
                      initial_reward = 500,
                      learning_rate = 1,
                      discount_factor = 0.9,
                      epsilon=_epsilon)
    
    if type(matrix)==pd.core.frame.DataFrame:
        Qlearn._qmatrix = matrix
        #onlyperform = True
    #display(Qlearn._qmatrix)
    rewardlist = ["http://192.168.1.68/timeclock/admin/groupdelete.php","http://192.168.1.68/timeclock/admin/index.php"]
    curl = ""
    
    #-=-=-=-=makedir=-=-=-=
    stateMap = defaultdict(factory)
    if os.path.exists(output):
        if statemap==None:
            shutil.rmtree(output)
            os.mkdir(output)
        else:
            stateMap = statemap
    else:
        os.mkdir(output)
    for code in os.listdir("./graphView/"):
        shutil.copyfile(os.path.join("./graphView/",code),os.path.join(output,code))
    #-=-=-==-=-=-=-=-=-=-=-
    if timebound:
        def some_task():
            global CLOSE
            time.sleep(activity_time)
            CLOSE=True
        t = Thread(target=some_task)
        t.start()
    
    
    startstate = chash.md5(env.get_Actions_OR_state())
    i_episode=0
    while(True):
        print("EPISODE= ",i_episode)
        if timebound:
            if CLOSE:
                break
        else:
            if i_episode>num_episodes:
                break
    #for i_episode in tqdm(range(num_episodes)):
        Qlearn._qmatrix.to_csv("./Q-table")
        save_stateMap(stateMap,"Q.map")
        env.reset(curl)
        curl = ""
        state = chash.md5(env.get_Actions_OR_state())
        state_actions = env.get_Actions_OR_state(True)
        prev_action = ""
        actionlist = []
        #startstate = state
        for t in itertools.count():
            urlbefore = ""
            #=-=--=-=-STATE GRAPH-=-=-=-=-
            env.website.save_screenshot(os.path.join(output,state+".png"))
            with open(os.path.join(output,state+".html"),"w") as htmlwriter:
                htmlwriter.write(env.website.page_source)
            #-=--=-=-=--=-=-=-=-=-=-=-=-=
            
            # Take a step
            action = Qlearn.get_best_action(state,state_actions,policy=False)
            #print("selected action ",action,"episode ",i_episode)
            elem = env.reverseEngineerAction(action)
            
            done=False
            reward = -99999
            if elem!=None:
                curl = env.website.current_url
                status,done = env.step(elem,login_url=login_urls,
                                       username=username,password=password,
                                       depth=depth)
                if status:
                    urlbefore = curl
                    currenturl = None
                    breakcount = 0
                    while currenturl==None and breakcount<5:
                        try:
                            currenturl = env.website.current_url
                            breakcount=6
                        except:
                            breakcount+=1
                            continue
                    if env.BaseURL in env.website.current_url:
                        #if env.website.current_url in rewardlist:
                        #    reward = 500
                        #    done=True
                        #    rewardlist.remove(env.website.current_url)
                        #else:
                        #    #reward = -1
                        #    Qlearn._qmatrix.loc[action,"count"]+=1
                        #    reward = 1/np.sum(Qlearn._qmatrix.loc[action,"count"])
                        Qlearn._qmatrix.loc[action,"count"]+=1
                        reward = 1/np.sum(Qlearn._qmatrix.loc[action,"count"])
                        curl = ""
                    else:
                        reward = -999999999999
                        done = True
                else:
                    print("status is none-=-",elem.get_attribute("outerHTML"))
                
            else:
                print("none---",len(state_actions),"selected=",action)
            if reward==-999999999999:
                next_state = state
                next_state_actions = state_actions
            else:
                S = env.get_Actions_OR_state()
                if S=='':
                    S = env.website.page_source
                    S,nodes = generateTree(S)
                    
                next_state = chash.md5(S)
                next_state_actions = env.get_Actions_OR_state(True)
                env.website.save_screenshot(os.path.join(output,next_state+".png"))
                with open(os.path.join(output,next_state+".html"),"w") as htmlwriter:
                    htmlwriter.write(env.website.page_source)

            if not onlyperform:
                if Qlearn.update_model(state, action, reward, next_state,next_state_actions)==-1:
                    done=True
                #-=-=-=-=makingGraph=-=-=-=-
                stateMap[state]["src"]=state
                stateMap[state]["edges"].append({"action":action,"state":next_state})
                if state==startstate:
                    stateMap[state]["start"] = 1
                stateMap[state]["url"] = urlbefore
                #-==-=-=-=-=-=-=-=-=-=-=-=-=
            prev_action = action

            # Update statistics
            #if not onlyperform:
            #    stats.episode_rewards[i_episode] += reward
            #    stats.episode_lengths[i_episode] = t

            if done==True:
                break

            state = next_state
            state_actions = next_state_actions
        #print(actionlist)
        i_episode+=1
    makeGraph(stateMap,output)
    #return stats,Qlearn,stateMap
    return 0,Qlearn,stateMap


# In[7]:


import argparse
from gooey import Gooey

@Gooey
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="The home page url of the website")
    parser.add_argument("--login_urls", help="The login page url of the website, add comma incase of multiple")
    parser.add_argument("--username", help="The username")
    parser.add_argument("--password", help="The password")
    parser.add_argument("--baseurl", help="The base url to check for out of scope websites(without Tailing /), Default is same as home url")
    parser.add_argument("--action_wait", help="Action weight time in seconds, default is 0.5",default=0.5,type=float)
    parser.add_argument("--episodes", help="Number of episodes to run, default in 2",default=2,type=int)
    parser.add_argument("--matrix", help="path of matrix to resume, default is None",default=None)
    parser.add_argument("--stateMap", help="path of statemap to resume, default is None",default=None)
    parser.add_argument("--timebound", help="To use time instead of episodes, default is False",action="store_true")
    parser.add_argument("--activity_time", help="max time to run the activity, default is 0",default=0,type=int)
    parser.add_argument("--depth", help="max valid actions in one episode, default is 100",default=100,type=int)
    
    args = parser.parse_args()
    
    
    #env = webEnv(url="http://192.168.1.68/timeclock/",BaseURL="http://192.168.1.68/timeclock",actionWait=0.5)
    login_urls = args.login_urls.split(",")
    base = ""
    if args.baseurl:
        base = args.baseurl
    else:
        base = args.url[::-1]
        
    env = webEnv(url=args.url,BaseURL=base,actionWait=args.action_wait)
    
    #stats,matrix,stateMap=q_learning(env, 2,timebound=True,activity_time=60)
    stats,matrix,stateMap=q_learning(env, args.episodes,timebound=args.timebound,activity_time=args.activity_time,
                                    matrix=args.matrix,statemap=args.stateMap,
                                    login_urls=login_urls,username=args.username,password=args.password,depth=args.depth)


# In[8]:

if __name__ == "__main__":
    main()


# In[ ]:




