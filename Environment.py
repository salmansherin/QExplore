"""
import cryptohash as chash
import random
import numpy as np
import pandas as pd
import wordninja
import nltk
import gensim
import enchant
import sister
from bs4 import BeautifulSoup
import string
from num2words import num2words
import matplotlib.pyplot as plt
import requests
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from prettytable import PrettyTable
import requests
import time
import exrex as ex
import js_regex
from sklearn.metrics.pairwise import cosine_similarity
import ast
import datetime



class webEnv:
    
    def __init__(self,url,BaseURL="http://localhost/timeclock/",actionWait=0.5):
        self.url = url
        self.tags_to_find = ['input','button','a','select']
        self.website  = webdriver.Firefox()
        self.website.get(url)
        self.datalabel = ['zipcode','city','streetname','secondaryaddress',
                'county','country','countrycode','state','stateabbr',
                'latitude','longitude','address','email','username',
                'password','sentence','word','paragraph','firstname',
                'lastname','fullname','age','phonenumber','date']
        self.embedding = sister.MeanEmbedding(lang="en")
        self.edict = enchant.Dict('en_US')
        self.tagAttr = {'a':[''],'button':['value','name'],
                        'select':['name','class'],
                   'input':['placeholder','name','value']}
        self.prev_password = None
        self.BaseURL = BaseURL
        self.currentDepth=0
        self.actionWait = actionWait
    
    def get_Actions_OR_state(self,action=False):
        tagstr = []
        for x in self.tags_to_find:
            xtags = self.website.find_elements_by_tag_name(x)
            if xtags!=[]:
                for element in xtags:
                    s = x+"!@!"
                    if x=='input':
                        if element.get_attribute('name')is not '' and element.get_attribute('name')is not None:
                            s+=element.get_attribute('name').strip()+'!@!'
                        else:
                            s+='nan!@!'
                        if element.get_attribute('value') is not '' and element.get_attribute('value') is not None:
                            s+=element.get_attribute('value').strip()+'!@!'
                            #s+='nan!@!'
                        else:
                            s+='nan!@!'
                        s+='nan'
                        
                    elif x=='button':
                        s+='nan!@!' #name
                        if element.get_attribute('value') is not '' and element.get_attribute('value') is not None:
                            s+=element.get_attribute('value').strip()+'!@!'
                        else:
                            s+='nan!@!'
                        s+='nan'
                    elif x=='a':
                        s+='nan!@!' #name
                        s+='nan!@!' #value
                        if element.get_attribute('href') is not '' and element.get_attribute('href') is not None:
                            s+=element.get_attribute('href').strip()
                        else:
                            s+='nan'
                    elif x=='select':
                        if element.get_attribute('name') is not '' and element.get_attribute('name') is not None:
                            s+=element.get_attribute('name').strip()+'!@!'
                        else:
                            s+='nan!@!'
                        s+='nan!@!' #value
                        s+='nan'    #href
                    tagstr.append(s)
        if action:
            return tagstr
        else:
            return "\n".join(tagstr)
        
    def reverseEngineerAction(self,action):
        tag,name,value,href = action.split('!@!')
        xpath='//'+tag+'['
        xpath2='//'+tag+'['
        att = []
        att2 = []
        if name!='nan':
            att.append('@name='+'"'+name+'"')
            att2.append('@name='+'"'+name+'"')
        if value!='nan':
            att.append('@value='+'"'+value+'"')
            att2.append('@value='+'"'+value+'"')
        if href!='nan':
            arrrr = href.split('/')
            if arrrr[-1]!='':
                att.append("@href="+"'"+arrrr[-1]+"'")
            else:
                att.append("@href="+"'"+arrrr[-2]+"'")
            att2.append("@href="+"'"+href+"'")
        xpath+=" and ".join(att)+"]"
        xpath2+=" and ".join(att2)+"]"
        try:
            return self.website.find_element_by_xpath(xpath)
        except:
            try:
                return self.website.find_element_by_xpath(xpath2)
            except:
                return None
        #return tag,name,value,href,xpath,xpath2
    
    #retrun a random string of length 8
    #def randomString(self):
        #return ''.join(random.choices(string.ascii_uppercase + string.digits, k = 8)) 
   
    def get_random_string(self,length):
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str

    def getvectors(self,sentences):
        vector = self.embedding(sentences)
        return vector
    
    def getsimilarity(self, feature_vec_1, feature_vec_2):    
        return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]
    
    def getdistance(self,a,b):
        return np.linalg.norm(a-b)
    
    def getSentence(self,html):
        soup = BeautifulSoup(html,'lxml')
        sentence = ""
        table = str.maketrans('', '', string.punctuation)
        for tag in self.tags_to_find:
            for t in soup.findAll(tag):
                att = t.attrs
                for chose in self.tagAttr[tag]:
                    try:
                        v = att[chose]
                        if type(v)==list:
                            sentence+=" ".join(v)+" "
                        else:
                            sentence+=' '+v
                    except:
                        continue
                
                sentence+=t.text
                #sentence = t.name+" "+sentence
        #print("before: ",sentence)
        #or cls in self.bootclasses:
        #   sentence = sentence.strip().replace(cls,'')
        #or cls in self.stopwords:
        #   sentence = sentence.strip().replace(cls,'')
            
        sentence = sentence.strip().translate(table)
        sentence = sentence.lower().replace('lastname','last-name')
        sentence = sentence.replace('firstname','first-name')
        sentence = sentence.replace('username','user-name')
        sentence = sentence.replace('userid','user-name')
        sentence = sentence.replace('enddate','end-date')
        sentence = sentence.replace('startdate','start-date')
        sentence = sentence.replace('cnic','identitynumber')
        #print("after class: ",sentence)
        sentence_new = ""
        for num in wordninja.split(sentence):
            word = ''.join([i for i in num if not i.isdigit()])
            try:
                if self.edict.check(word) and len(word)>1:
                    sentence_new+=word+" "
                    #print(word)
            except:
                pass
        #print("after ninja: ",sentence_new)
        return " ".join(list(set(sentence_new.split(" "))))
    
    #This method execute each element of the DOM depending on the type of element
    
    def click(self,elem):
        try:
            elem.click()
            time.sleep(self.actionWait)
            return 1
        except:
            return 0        
    
    def write(self,elem,login_url,username,password):
        html = elem.get_attribute("outerHTML")
        sentence = self.getSentence(html)
        if sentence!="":
            v_sentence = self.getvectors(sentence)
            #print("***********************")
            simL = []
            for x in self.datalabel:
                x_vector = self.getvectors(x)
                sim = self.getsimilarity(x_vector,v_sentence)
                simL.append(sim)
                #print(x+" is "+str(sim)+" similar to '"+sentence+"'")
            mostsim = self.datalabel[np.argmax(simL)]
            #print("'"+sentence+"' is most similar to "+mostsim)
            PARAMS = {'value':mostsim}
            r = requests.get(url = "http://localhost:3000/", params = PARAMS)
            d = ast.literal_eval(r.text.replace("`",""))
            if mostsim=='date':
                date = d["'d'"][0].split("T")[0]
                date = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%m/%d/%Y')
                d["'d'"][0]=date
            if mostsim=="paragraph" or mostsim=="word":
                d["'d'"][0] = "abcd123456"
            if mostsim=="username" and self.website.current_url in login_url:
                d["'d'"][0] = username
            if mostsim=="password":
                if self.website.current_url in login_url:
                    d["'d'"][0] = password
                else:
                    if self.prev_password!=None:
                        d["'d'"][0] = self.prev_password
                    else:
                        self.prev_password = d["'d'"][0]

            #print("Generated: ",d["'d'"][0])
            try:
                if elem.get_attribute("value")!=d["'d'"][0]:
                    elem.send_keys(d["'d'"][0])
                return 1
            except:
                return 0
        else:
            PARAMS = {'value':'word'}
            r = requests.get(url = "http://localhost:3000/", params = PARAMS)
            d = ast.literal_eval(r.text.replace("`",""))
            try:
                elem.send_keys(d["'d'"][0])
                return 1
            except:
                return 0
               
    def checkDone(self,depth):
        if self.currentDepth>=depth:
            return True
        else:
            return False
    
    
    def step(self,elem,login_url="",username=None,password=None,depth=4):
        
        clickable = ["a","button","submit","select","radio","checkbox"]
        writable = ["input","text","password","search"]
        status = 0
        
        if elem.tag_name in clickable or elem.get_attribute('Type') in clickable:
            if elem.tag_name=="select":
                select = Select(elem)
                option = random.choice(select.options)
                status = self.click(option)                    
            else:
                status = self.click(elem)
                
            if status:
                self.currentDepth+=1
                
        elif elem.tag_name in writable or elem.get_attribute('Type') in writable:
            status = self.write(elem,login_url,username,password)
            if status:
                self.currentDepth+=1
        else:
            return 0,self.checkDone(depth)
        
        return status,self.checkDone(depth)

    def reset(self):
        self.website.get(self.url)
        self.currentDepth=0
        self.prev_password = None
        
    def close(self):
        try:
            self.website.close()
            self.website.close()
        except:
            pass
"""

import cryptohash as chash
import random
import numpy as np
import pandas as pd
import wordninja
import nltk
import gensim
import enchant
import sister
from bs4 import BeautifulSoup
import string
from num2words import num2words
import matplotlib.pyplot as plt
import requests
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from prettytable import PrettyTable
import requests
import time
import exrex as ex
import js_regex
from sklearn.metrics.pairwise import cosine_similarity
import ast
import datetime



class webEnv:
    
    def __init__(self,url,BaseURL="http://localhost/timeclock/",actionWait=0.5):
        self.url = url
        self.tags_to_find = ['input','button','a','select']
        self.website  = webdriver.Firefox()
        self.website.get(url)
        self.datalabel = ['zipcode','city','streetname','secondaryaddress',
                'county','country','countrycode','state','stateabbr',
                'latitude','longitude','address','email','username',
                'password','sentence','word','paragraph','firstname',
                'lastname','fullname','age','phonenumber','date']
        self.embedding = sister.MeanEmbedding(lang="en")
        self.edict = enchant.Dict('en_US')
        self.tagAttr = {'a':[''],'button':['value','name'],
                        'select':['name','class'],
                   'input':['placeholder','name','value']}
        self.prev_password = None
        self.BaseURL = BaseURL
        self.currentDepth=0
        self.actionWait = actionWait
    
    def get_Actions_OR_state(self,action=False):
        tagstr = []
        for x in self.tags_to_find:
            xtags = self.website.find_elements_by_tag_name(x)
            if xtags!=[]:
                for element in xtags:
                    s = x+"!@!"
                    if x=='input':
                        try:
                            if element.get_attribute('name')is not '' and element.get_attribute('name')is not None:
                                s+=element.get_attribute('name').strip()+'!@!'
                            else:
                                s+='nan!@!'
                        except:
                            s+='nan!@!'
                        try:
                            value_attr = element.get_attribute('value')
                            type_attr = element.get_attribute('type')
                        
                            if value_attr is not '' and value_attr is not None and type_attr not in ["text","password"]:
                                s+=element.get_attribute('value').strip()+'!@!'
                                #s+='nan!@!'
                            else:
                                s+='nan!@!'
                        except:
                            s+='nan!@!'
                        s+='nan'
                        
                    elif x=='button':
                        s+='nan!@!' #name
                        try:
                            if element.get_attribute('value') is not '' and element.get_attribute('value') is not None:
                                s+=element.get_attribute('value').strip()+'!@!'
                            else:
                                s+='nan!@!'
                        except:
                            s+='nan!@!'
                        s+='nan'
                    elif x=='a':
                        s+='nan!@!' #name
                        s+='nan!@!' #value
                        try:
                            if element.get_attribute('href') is not '' and element.get_attribute('href') is not None:
                                s+=element.get_attribute('href').strip()
                            else:
                                s+='nan'
                        except:
                            s+='nan'
                    elif x=='select':
                        try:
                            if element.get_attribute('name') is not '' and element.get_attribute('name') is not None:
                                s+=element.get_attribute('name').strip()+'!@!'
                            else:
                                s+='nan!@!'
                        except:
                            s+='nan!@!'
                        s+='nan!@!' #value
                        s+='nan'    #href
                    tagstr.append(s)
        if action:
            dedup_list = []
            for i in tagstr:
                if i not in dedup_list:
                    dedup_list.append(i)

            #tagstr = list(set(tagstr))
            #tagstr.sort(reverse=True)
            return dedup_list
        else:
            for i in range(len(tagstr)):
                x = tagstr[i].split("!@!")
                if x[0]=="a":
                    x[-1]="#"
                x = "!@!".join(x)
                tagstr[i]=x
            return "\n".join(tagstr)
    """    
    def reverseEngineerAction(self,action):
        tag,name,value,href = action.split('!@!')
        xpath='//'+tag+'['
        xpath2='//'+tag+'['
        att = []
        att2 = []
        if name!='nan':
            att.append('@name='+'"'+name+'"')
            att2.append('@name='+'"'+name+'"')
        if value!='nan':
            att.append('@value='+'"'+value+'"')
            att2.append('@value='+'"'+value+'"')
        if href!='nan':
            arrrr = href.split('/')
            if arrrr[-1]!='':
                att.append("@href="+"'"+arrrr[-1]+"'")
            else:
                att.append("@href="+"'"+arrrr[-2]+"'")
            att2.append("@href="+"'"+href+"'")
        xpath+=" and ".join(att)+"]"
        xpath2+=" and ".join(att2)+"]"
        try:
            return self.website.find_element_by_xpath(xpath),xpath2
        except:
            try:
                return self.website.find_element_by_xpath(xpath2),xpath
            except:
                return None
    """
    def reverseEngineerAction(self,action):
        tag,name,value,href = action.split('!@!')
        xpath='//'+tag+'['
        att = []
        hreflist = []
        XPATH = ""
        if name!='nan':
            att.append('@name='+'"'+name+'"')
        if value!='nan':
            att.append('@value='+'"'+value+'"')
        if href!='nan':
            hreflist.append(href)
            for x in range(1,len(href.split("/"))):
                temphref = "/".join(href.split("/")[x::])
                hreflist.append(temphref)
                hreflist.append("../"+temphref)
                hreflist.append("/"+temphref)
        elem = None
        if hreflist==[]:
            try:
                XPATH = xpath+" and ".join(att)+"]"
                elem = self.website.find_element_by_xpath(XPATH)
            except:
                pass
        else:
            for x in hreflist:
                try:
                    _att = []
                    _att.extend(att)
                    _att.append("@href="+"'"+x+"'")
                    XPATH = xpath+" and ".join(_att)+"]"
                    elem = self.website.find_element_by_xpath(XPATH)
                    break
                except:
                    continue
            if elem==None:
                #print("none")
                try:
                    x="#"
                    _att = []
                    _att.extend(att)
                    _att.append("@href="+"'"+x+"'")
                    XPATH = xpath+" and ".join(_att)+"]"
                    elem = self.website.find_element_by_xpath(XPATH)
                except:
                    pass
        return elem
            

    def get_random_string(self,length):
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str

    def getvectors(self,sentences):
        vector = self.embedding(sentences)
        return vector
    
    def getsimilarity(self, feature_vec_1, feature_vec_2):    
        return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]
    
    def getdistance(self,a,b):
        return np.linalg.norm(a-b)
    
    def getSentence(self,html):
        soup = BeautifulSoup(html,'lxml')
        sentence = ""
        table = str.maketrans('', '', string.punctuation)
        for tag in self.tags_to_find:
            for t in soup.findAll(tag):
                att = t.attrs
                for chose in self.tagAttr[tag]:
                    try:
                        v = att[chose]
                        if type(v)==list:
                            sentence+=" ".join(v)+" "
                        else:
                            sentence+=' '+v
                    except:
                        continue
                
                sentence+=t.text
                #sentence = t.name+" "+sentence
        #print("before: ",sentence)
        #or cls in self.bootclasses:
        #   sentence = sentence.strip().replace(cls,'')
        #or cls in self.stopwords:
        #   sentence = sentence.strip().replace(cls,'')
            
        sentence = sentence.strip().translate(table)
        sentence = sentence.lower().replace('lastname','last-name')
        sentence = sentence.replace('firstname','first-name')
        sentence = sentence.replace('username','user-name')
        sentence = sentence.replace('userid','user-name')
        sentence = sentence.replace('enddate','end-date')
        sentence = sentence.replace('startdate','start-date')
        sentence = sentence.replace('cnic','identitynumber')
        #print("after class: ",sentence)
        sentence_new = ""
        for num in wordninja.split(sentence):
            word = ''.join([i for i in num if not i.isdigit()])
            try:
                if self.edict.check(word) and len(word)>1:
                    sentence_new+=word+" "
                    #print(word)
            except:
                pass
        #print("after ninja: ",sentence_new)
        return " ".join(list(set(sentence_new.split(" "))))
    
    #This method execute each element of the DOM depending on the type of element
    
    def click(self,elem):
        try:
            elem.click()
            time.sleep(self.actionWait)
            return 1
        except:
            return 0        
    
    def write(self,elem,login_url,username,password,email):
        html = elem.get_attribute("outerHTML")
        sentence = self.getSentence(html)
        if sentence!="":
            v_sentence = self.getvectors(sentence)
            #print("***********************")
            simL = []
            for x in self.datalabel:
                x_vector = self.getvectors(x)
                sim = self.getsimilarity(x_vector,v_sentence)
                simL.append(sim)
                #print(x+" is "+str(sim)+" similar to '"+sentence+"'")
            mostsim = self.datalabel[np.argmax(simL)]
            #print("'"+sentence+"' is most similar to "+mostsim)
            PARAMS = {'value':mostsim}
            r = requests.get(url = "http://localhost:3000/", params = PARAMS)
            d = ast.literal_eval(r.text.replace("`",""))
            if mostsim=='date':
                date = d["'d'"][0].split("T")[0]
                date = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%m/%d/%Y')
                d["'d'"][0]=date
            if mostsim=="paragraph" or mostsim=="word":
                d["'d'"][0] = "abcd123456"
            if mostsim=="username":
                if self.website.current_url in login_url:
                    if username!=None:
                        d["'d'"][0] = username
                else:
                    if username!=None:
                        d["'d'"][0] = username
                    print(self.website.current_url,login_url,self.website.current_url in login_url)
            if mostsim=="email":
                if self.website.current_url in login_url:
                    if email!=None:
                        d["'d'"][0] = email
                else:
                    print(self.website.current_url,login_url,self.website.current_url in login_url)
            if mostsim=="password":
                if self.website.current_url in login_url:
                    if password!=None:
                        d["'d'"][0] = password
                else:
                    if password!=None:
                        d["'d'"][0] = password
                        
                    if self.prev_password!=None:
                        d["'d'"][0] = self.prev_password
                    else:
                        self.prev_password = d["'d'"][0]

            #print("Generated: ",d["'d'"][0]," mostsim=",mostsim)
            try:
                if elem.get_attribute("value")=="" or elem.get_attribute("value")==None:
                    if elem.get_attribute("value")!=d["'d'"][0]:
                        elem.send_keys(d["'d'"][0])
                return 1
            except:
                return 0
        else:
            PARAMS = {'value':'word'}
            r = requests.get(url = "http://localhost:3000/", params = PARAMS)
            d = ast.literal_eval(r.text.replace("`",""))
            try:
                if elem.get_attribute("value")=="" or elem.get_attribute("value")==None:
                    if elem.get_attribute("value")!=d["'d'"][0]:
                        elem.send_keys(d["'d'"][0])
                return 1
                #elem.send_keys(d["'d'"][0])
                #return 1
            except:
                return 0
               
    def checkDone(self,depth):
        if self.currentDepth>=depth:
            return True
        else:
            return False
    
    
    def step(self,elem,login_url="",username=None,password=None,depth=4,email=None):
        
        clickable = ["a","button","submit","select","radio","checkbox","image"]
        writable = ["input","text","password","search"]
        status = 0
        
        if elem.tag_name in clickable or elem.get_attribute('Type') in clickable:
            if elem.tag_name=="select":
                try:
                    select = Select(elem)
                    option = random.choice(select.options)
                    status = self.click(option)
                except:
                    status = 0
            else:
                status = self.click(elem)
                
            if status:
                self.currentDepth+=1
                
        elif elem.tag_name in writable or elem.get_attribute('Type') in writable:
            status = self.write(elem,login_url,username,password,email)
            if status:
                self.currentDepth+=1
        else:
            return 0,self.checkDone(depth)
        
        return status,self.checkDone(depth)

    def reset(self,curl=""):
        if curl=="":
            self.website.get(self.url)
            self.currentDepth=0
            self.prev_password = None
        else:
            self.website.get(curl)
            #self.currentDepth=0
            #self.prev_password = None
        
    def close(self):
        try:
            self.website.close()
            self.website.close()
        except:
            pass