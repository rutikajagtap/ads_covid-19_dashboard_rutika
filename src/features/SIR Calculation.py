#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

from datetime import datetime

import seaborn as sns
sns.set(style="darkgrid")

from scipy import optimize
from scipy import integrate

import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go

from ipywidgets import widgets, interactive

import requests
from bs4 import BeautifulSoup

df_raw_infected = pd.read_csv('...\\data\\raw\\COVID-19\\csse_covid_19_data\\csse_covid_19_time_series\\time_series_covid19_confirmed_global.csv')
time_idx = df_raw_infected.columns[4:]

df_infected = pd.DataFrame({
    'date':time_idx})

country_list = df_raw_infected['Country/Region'].unique()

for each in country_list:
    df_infected[each]=np.array(df_raw_infected[df_raw_infected['Country/Region']==each].iloc[:,4:].sum(axis=0))
    
df_infected = df_infected.iloc[60:]

def SIR_model_t(SIR,t,beta,gamma):

    S,I,R=SIR
    dS_dt=-beta*S*I/N0          
    dI_dt=beta*S*I/N0-gamma*I
    dR_dt=gamma*I
    return dS_dt,dI_dt,dR_dt


def fit_odeint(x, beta, gamma):
    return integrate.odeint(SIR_model_t, (S0, I0, R0), t, args=(beta, gamma))[:,1] 

page = requests.get("https://www.worldometers.info/world-population/population-by-country/")
soup = BeautifulSoup(page.content, 'html.parser')
html_table= soup.find('table')
all_rows= html_table.find_all('tr')
final_data_list=[]

for pos,rows in enumerate(all_rows):
    col_list=[each_col.get_text(strip=True) for each_col in rows.find_all('td')]
    final_data_list.append(col_list)
    
population = pd.DataFrame(final_data_list).dropna().rename(columns={0:'index', 1:'country', 2:'population', 3:'a', 4:'b', 5:'c', 6:'d', 7:'e', 8:'f', 9:'g', 10:'h', 11:'i'})
population = population.drop(['index','a','b','c','d','e','f','g','h','i'],axis=1)
population['population'] = population['population'].str.replace(',','')
population['population'] = population['population'].apply(float)
population = population.set_index('country')

df_country = pd.DataFrame({'country':country_list}).set_index('country')
df_analyze = pd.merge(df_country,population,left_index=True, right_on='country',how='left')

df_analyze = df_analyze.replace(np.nan,1000000).T
df_analyze.iloc[0].apply(float)


for each in country_list:
    ydata = np.array(df_infected[each])
    t = np.arange(len(ydata))
    I0=ydata[0]
    N0=np.array(df_analyze[each])
    N0 = N0.astype(np.float64)
    S0=N0-I0
    R0=0
    
    popt, pcov = optimize.curve_fit(fit_odeint, t, ydata, maxfev = 1200)
    perr = np.sqrt(np.diag(pcov))
    
    fitted = fit_odeint(t, *popt)
    df_infected[each + '_SIR'] = fitted
    
df_infected = df_infected.drop(['date'],axis=1)
for each in country_list:
    df_infected = df_infected.drop([each], axis=1)
    
df_infected.to_csv('...\\data\\processed\\COVID_SIR.csv' , sep=';', index=False)

