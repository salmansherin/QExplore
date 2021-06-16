from multiprocessing import Pool
import time
from tqdm import *
from apted import APTED
from apted.helpers import Tree
import pandas as pd
import numpy as np
import glob, os
import itertools
from bs4 import BeautifulSoup
import bs4

from collections import OrderedDict, Callable, defaultdict

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


def makeTree(file):
    with open(file,"r") as f:
        S = f.read()
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

def generateTree(file):
    html = makeTree(file)
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

def calculateDomDiveristy(X):
    tree1,n1 = generateTree(X[0])
    tree2,n2 = generateTree(X[1])
    t1 = Tree.from_text(tree1)
    t2 = Tree.from_text(tree2)
    apted = APTED(t1, t2)
    ted = apted.compute_edit_distance()
    DD = ted/max(n1,n2)
    return DD,X,ted


if __name__ == '__main__':
    c = input("Enter choice Q or C:").lower()
    if c=="q":
        path = "./Q_Result/"
        exclude = [path+"index.html",path+"temp.html"]
        states = [file for file in glob.glob(path+"*.html") if file not in exclude]
    elif c=="c":
        path = "./doms/"
        exclude = [path+"temp.html"]
        states = [file for file in glob.glob(path+"*.html") if file not in exclude]
    else:
        print("wrong choice")
        exit(1)
    
    val = []
    
    def lol():
        return -1
    Q = defaultdict(lol)
    
    with Pool() as p:
        max_ = len(states)
        with tqdm(total=max_) as pbar:
            pairs = [x for x in itertools.product(states,states) if x[0]!=x[1]]
            blacklist = []
            T = []
            for x,y in pairs:
                if (x,y) not in blacklist:
                    T.append((x,y))
                blacklist.append((y,x))
            for i, v in enumerate(p.imap_unordered(calculateDomDiveristy, T)):
                val.append(v[0])
                if Q["--".join(v[1])]==-1:
                    if v[2]<=3:
                        Q["--".join(v[1])] = 1
                        Q["--".join(v[1][::-1])] = 2
                pbar.update()
            print("\nDom Diversity: ",np.mean(val))
    
    blacklistQ = set()
    statesQU = set()
    for x in Q.keys():
        p = x.split("--")
        if Q[x]==1:
            blacklistQ.add(p[1])
        elif Q[x]==2:
            blacklistQ.add(p[0])
    for x in Q.keys():
        p = x.split("--")
        if p[0] not in blacklistQ:
            statesQU.add(p[0])
        if p[1] not in blacklistQ:
            statesQU.add(p[1])
    print("DOM uniquness: ",1-(len(statesQU)/len(states)))
    with open(c+"-diversity.txt","w") as ans:
        ans.write(str(np.mean(val))+"\n")
        ans.write(str(1-(len(statesQU)/len(states)))+"\n")
