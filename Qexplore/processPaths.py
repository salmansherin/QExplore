from multiprocessing import Pool
import time
from tqdm import *
import pandas as pd
import numpy as np
import glob, os
import itertools
from bs4 import BeautifulSoup
import bs4
import json

from collections import OrderedDict, Callable, defaultdict

def getpathSimilarity(path):
    A = set(path[0])
    B = set(path[1])
    shared = len(A.intersection(B))
    pathsim = (2*shared)/(len(A)+len(B))
    return pathsim

def calculate_Average_NavigationalDiversity(data):
    keys = [key for key in data.keys() if data[key]!=[]]
    diversity = []
    for key in tqdm(keys):
        paths = data[key]
        sim = []
        for path1,path2 in product(paths,paths):
            sim.append(getpathSimilarity(path1,path2))
        sim = np.array(sim)
        if np.sum(sim==1)==sim.shape[0]:
            diversity.append(1)
        else:
            diversity.append(1-np.min(sim))
    return np.mean(diversity)


if __name__ == '__main__':
    
    with Pool() as p:
        with open("qpaths.json") as json_file:
            data = json.load(json_file)
        max_ = len(data.keys())
        with tqdm(total=max_) as pbar:
            diversity = []
            for key in data.keys():
                if data[key]!=[]:
                    paths = data[key]
                    sim = []
                    pairs = itertools.product(paths,paths)
                    for i, v in enumerate(p.imap_unordered(getpathSimilarity, pairs)):
                        sim.append(v)
                    sim = np.array(sim)
                    if np.sum(sim==1)==sim.shape[0]:
                        diversity.append(1)
                    else:
                        diversity.append(1-np.min(sim))
                pbar.update()
            umt = np.mean(diversity)
            print("Q-Div",umt)
         
        with open("cpaths.json") as json_file:
            data = json.load(json_file)
        max_ = len(data.keys())
        with tqdm(total=max_) as pbar:
            diversity = []
            for key in data.keys():
                if data[key]!=[]:
                    paths = data[key]
                    sim = []
                    if len(paths)<=1:
                        diversity.append(0)
                    else:
                        pairs = itertools.product(paths,paths)
                        for i, v in enumerate(p.imap_unordered(getpathSimilarity, pairs)):
                            sim.append(v)
                        sim = np.array(sim)
                        if np.sum(sim==1)==sim.shape[0]:
                            diversity.append(1)
                        else:
                            diversity.append(1-np.min(sim))
                    #print(diversity)
                pbar.update()
            umt2 = np.mean(diversity)
            print("C-Div",umt2)
        with open("Nav-Div.txt","w") as file:
            file.write(str(umt)+"\n")
            file.write(str(umt2))