#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import linear_model

reg = linear_model.LinearRegression(fit_intercept=True)
import pandas as pd

from scipy import signal


def get_doubling_time_via_regression(in_array):
    
    y= np.array(in_array)
    X= np.arange(-1,2).reshape(-1, 1)
    
    assert len(in_array)==3 
    reg.fit(X,y)
    intercept = reg.intercept_
    slope = reg.coef_
    
    return intercept/slope



def savgol_filter(df_input,column='confirmed',window=5):
    window=5,
    degree=1
    df_result=df_input
    
    filter_in=df_input[column].fillna(0)
    
    result=signal.savgol_filter(np.array(filter_in),
                                5,
                                1)
    
    df_result[column+'_filtered']=result
    return df_result



def rolling_reg(df_input,column='confirmed'):
    
    days_back=3
    result=df_input[column].rolling(
                window=days_back,
                min_periods=days_back).apply(get_doubling_time_via_regression, raw=False)
    return result



def calc_filtered_data(df_input,filter_on='confirmed'):
    
    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns))
    
    pd_filtered_result=df_input[['state','country',filter_on]].groupby(['state','country']).apply(savgol_filter)
    pd_filtered_result=pd_filtered_result.reset_index()
    df_output=pd.merge(df_input,pd_filtered_result[['index',filter_on+'_filtered']],on=['index'],how='left')
    
    return df_output



def calc_doubling_rate(df_input,filter_on='confirmed'):
    
    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns))
    
    pd_DR_result=df_input.groupby(['state','country']).apply(rolling_reg,filter_on)
    pd_DR_result=pd_DR_result.reset_index()
    pd_DR_result=pd_DR_result.rename(columns={filter_on:filter_on + '_DR',
                                                        'level_2':'index'})
    
    df_output=pd.merge(df_input,pd_DR_result[['index',filter_on+'_DR']],on=['index'],how='left')
    
    return df_output



if __name__=='__main__':
    test_data_reg=np.array([2,4,6])
    result=get_doubling_time_via_regression(test_data_reg)
    print('The test slope is: ' + str(result))
    
    pd_JH_data=pd.read_csv('...\\data\\processed\\COVID_relational_confirmed.csv' , sep=';', parse_dates=[0])
    pd_JH_data=pd_JH_data.sort_values('date',ascending=True).reset_index().copy()
    
    pd_result_larg=calc_filtered_data(pd_JH_data)
    pd_result_larg=calc_doubling_rate(pd_result_larg)
    pd_result_larg=calc_doubling_rate(pd_result_larg,'confirmed_filtered')
    pd_result_larg.to_csv('...\\data\\processed\\COVID_relational_confirmed_final.csv' , sep=';', index=False)
    print(pd_result_larg.head())


# In[ ]:




