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

df_infected = pd.read_csv('...\\data\\processed\\COVID_SIR.csv' , sep=';')
    
def multi_plot(df, title, addAll = True):
    fig = go.Figure()

    for column in df.columns.tolist():
        fig.add_trace(
            go.Scatter(
                x = df.index,
                y = df[column],
                name = column
            )
        )

    button_all = dict(label = 'All',
                      method = 'update',
                      args = [{'visible': df.columns.isin(df.columns),
                               'title': 'All',
                               'showlegend':True}])

    def create_layout_button(column):
        return dict(label = column,
                    method = 'update',
                    args = [{'visible': df.columns.isin([column]),
                             'title': column,
                             'showlegend': True}])

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 0,
            buttons = ([button_all] * addAll) + list(df.columns.map(lambda column: create_layout_button(column)))
            )
        ],
         yaxis_type="log"       
    )
    # Update remaining layout properties
    fig.update_layout(
        title_text=title,
        height=800
        
    )
   
    fig.show()
    
multi_plot(df_infected, title="Logarithmic COVID-19 time series total infected by SIR model")    

