#!/usr/bin/env python
# coding: utf-8

# In[1]:


def getWeights(d,size):
    '''
    d:fraction
    k:the number of samples
    w:weight assigned to each samples
    
    '''
    # thres>0 drops insignificant weights
    w=[1.]
    for k in range(1,size):
        w_ = -w[-1]/k*(d-k+1)
        w.append(w_)
    w=np.array(w[::-1]).reshape(-1,1) #sort and reshape the w
    return w


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['figure.dpi'] = 300
import yfinance as yf
def get_data(symbols, begin_date=None,end_date=None):
    df = yf.download('AAPL', start = '2000-01-01',
                     auto_adjust=True,#only download adjusted data
                     end= '2010-12-31') 
    #my convention:always lowercase
    df.columns = ['open','high','low',
                  'close','volume'] 
    
    return df
Apple_stock = get_data('AAPL', '2000-01-01', '2010-12-31')   
price = Apple_stock['close']
price.head()


# In[6]:


def weight_by_d(dRange=[0,1], nPlots=11, size=6):
    '''
    dRange: the range of d
    nPlots: the number of d we want to check
    size: the data points used as an example
    w: collection of w by different d value
    '''
    
    w=pd.DataFrame()
    
    for d in np.linspace(dRange[0],dRange[1],nPlots):
        w_=getWeights(d,size=size)
        w_=pd.DataFrame(w_,index=range(w_.shape[0])        [::-1],columns=[d])
        w=w.join(w_,how='outer')
        
    return w
weight_by_d = weight_by_d()
weight_by_d


# In[11]:


def get_skip(series,d = 0.1,thres=.01):
    '''
    This part is independent of stock price data.
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    
    #1) Compute weights for the longest series
    w=getWeights(d,series.shape[0])
    #2) Determine initial calcs to be skipped based on weight-loss threshold
    w_=np.cumsum(abs(w))
    w_/=w_[-1]
    skip=w_[w_>thres].shape[0]
    
    return skip


# In[12]:


def transfor_data_by_frac_diff(col,d = 0.1,thres=.01):
    #3) Apply weights to values
    df = pd.Series()
    skip = get_skip(col)
    
    for i in range(skip, col.shape[0]):
        i_index = col.index[i]
        data = np.dot(w[-(i+1):,:].T, col.loc[:i_index])[0]
        
        df[i_index] = data
                   
    return df


# In[13]:


def trans_a_bunch_of_data(df,d = 0.1,thres=.01):
    a_bunch_of_trans_data = pd.DataFrame()
    
    for col in df.columns:
        trans_data = transfor_data_by_frac_diff(df[col]           ,d =   d,thres=thres)
        a_bunch_of_trans_data[col] = trans_data
    
    return a_bunch_of_trans_data


# In[14]:


def getWeights_FFD(d=0.1, thres=1e-5):
    
    w,k=[1.],1
    while True:
        w_=-w[-1]/k*(d-k+1)
        if abs(w_)<thres:break
        w.append(w_)
        k+=1
    return np.array(w[::-1]).reshape(-1,1)


# In[15]:


w_FFD = getWeights_FFD(thres=1e-4)
w_FFD.shape


# In[27]:


def transfer_data_by_frac_diff_FFD(col, d=0.1, thres=1e-4):
    #3) Apply weights to values
    w=getWeights_FFD(d,thres)
    width=len(w)-1
    
    df = pd.Series()
    #widow size can't be larger than the size of data
    if width >= col.shape[0]:raise Exception("width is oversize")
        
    for i in range(width, col.shape[0]):
        i_0_index, i_1_index = col.index[i-width], col.index[i]
        data = np.dot(w.T, col.loc[i_0_index:i_1_index])[0]
        
        df[i_1_index] = data
                   
    return df


# In[28]:


price_trans = transfer_data_by_frac_diff_FFD(price)
price_trans.shape, price.shape


# In[29]:


def trans_a_bunch_of_data_FFD(df, d=0.1, thres=1e-4):
    a_bunch_of_trans_data = pd.DataFrame()
    
    for col in df.columns:
        trans_data = transfer_data_by_frac_diff_FFD(df[col],                      d=d, thres=thres)
        a_bunch_of_trans_data[col] = trans_data
    
    return a_bunch_of_trans_data
trans_a_bunch_of_data_FFD(Apple_stock, d=0.1, thres=1e-4)


# In[30]:


from statsmodels.tsa.stattools import adfuller
def get_adf_corr():
    out=pd.DataFrame(columns=['adfStat','pVal','lags',                             'nObs','95% conf','corr'])
    price_log = np.log(price)
    for d in np.linspace(0, 1 , 11):
        price_trans = transfer_data_by_frac_diff_FFD(                       price_log, d=d, thres=1e-4)
        corr = price_corr_np = np.corrcoef(price.loc[price_trans.index], price_trans)[0,1]
        adf=adfuller(price_trans, maxlag=1, regression='c',autolag=None)
        out.loc[d]=list(adf[:4])+[adf[4]['5%']]+[corr] 
        # with critical value
        
    return out


# In[31]:


out = get_adf_corr()
out


# In[36]:


ax1 = out['corr'].plot(figsize=(10, 6), color='r')
ax2 = out['adfStat'].plot(secondary_y=True, fontsize=20, color='g', ax=ax1)
ax1.set_title('AdfStat and Corr', fontsize=14)
ax1.set_xlabel('d value', fontsize=12)
ax1.set_ylabel('corr', fontsize=12)
ax2.set_ylabel('adfStat', fontsize=12)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='lower left', fontsize=12)
plt.axhline(out['95% conf'].mean(),linewidth=3, color='m',linestyle='--');


# In[ ]:




