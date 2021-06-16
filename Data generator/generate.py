import requests
import ast
from prettytable import PrettyTable


data = ['zipcode','city','streetname','secondaryaddress',
        'county','country','countrycode','state','stateabbr',
        'latitude','longitude','address','email','username',
        'password','sentence','word','paragraph','firstname',
        'lastname','name','age','phonenumber','date']

def table(data):
    t = PrettyTable(['Index', 'value'])
    for i,x in enumerate(data):
        t.add_row([str(i),x])
    t.add_row(["-1","Exit"])
    print(t)



import requests
import ast
URL = "http://localhost:3000/"
while(True):
    table(data)
    breaker = int(input("enter required index "))
    if breaker==-1:
        break
    PARAMS = {'value':data[breaker]}
    r = requests.get(url = URL, params = PARAMS)
    d = ast.literal_eval(r.text.replace("`",""))
    print("Generated: ",d["'d'"][0])
