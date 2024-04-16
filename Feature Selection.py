#Importing Libraries
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input,Dense, Dropout
from tensorflow.keras.models import Model
import joblib
from joblib import dump, load
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import matplotlib.pyplot as plt 
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import Matern,RationalQuadratic,DotProduct,ExpSineSquared
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle


#% Feature Selection for CO2
df = pd.read_csv('Dataset.csv')
loading_data = df['CO2 Loading'].unique()
MEA_data = df['MEA Concentration'].unique()
df_new = pd.DataFrame()
max_wn = df['Wavenumber'].max()
min_wn = df['Wavenumber'].min()
start = round(min_wn,0)
interval = 100
end = start+interval
col = ['MEA Concentration']

while end <= max_wn:
    df1 = df[(df['Wavenumber'].between(start,end))]
    text = str(start)+'-'+str(end)
    col.append(text) 
    for i in range(len(loading_data)):
        for j in range(len(MEA_data)):
                
            df2 = df1[df1['CO2 Loading']==loading_data[i]]
            df2 = df2[df2['MEA Concentration']==MEA_data[j]]
            df2 = df2.mean()
       
            df_new = pd.concat([df_new,df2],axis=1)
                
    start = start+interval
    end = start+interval
    
df_new = df_new.T

wavenumber_data = df_new['Wavenumber'].unique()
df2 = pd.DataFrame()
for i in range(len(wavenumber_data)):
    df1 = pd.DataFrame(df_new[df_new['Wavenumber']==wavenumber_data[i]].iloc[:,2].values)
    df2 = pd.concat([df2,df1],axis=1)

df_new = df_new.iloc[:len(loading_data)*len(MEA_data),:].values

X = df_new[:,1]
X = np.reshape(X,(-1,1))
Y = df_new[:,-1]
Y = np.reshape(Y,(-1,1))

df2 = df2.iloc[:,:].values
df_new = np.concatenate([X,df2,Y],axis=1)

X = df_new[:,:-1]
Y = df_new[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3) 

fs = SelectKBest(score_func=mutual_info_regression, k=3)
fs.fit(X_train, y_train)
X_fs = fs.transform(X)
X_fs = np.concatenate([df_new[:,0].reshape((-1,1)),X_fs],axis=1)

fig1 = plt.figure(dpi=400)
ax1 = fig1.add_axes([0,0,1,1])
ax1.bar(np.arange(1,len(col)+1,1),fs.scores_)
ax1.set_xticks(np.arange(1,len(col)+1,1),col,rotation=90)
ax1.set_ylabel('Mutual Information Regression Score')
ax1.set_xlabel('Features')
ax1.set_yticks(np.arange(0.0,1.2,0.2))
ax1.set_yticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'])
fig1.tight_layout()
score = fs.scores_

#%% Feature Selection for CO2

df_new = pd.DataFrame()
start = round(min_wn,0)
end = start+interval
col = ['CO2 Loading']

while end <= max_wn:
    df1 = df[(df['Wavenumber'].between(start,end))]
    text = str(start)+'-'+str(end)
    col.append(text) 
    for i in range(len(loading_data)):
        for j in range(len(MEA_data)):
                
            df2 = df1[df1['CO2 Loading']==loading_data[i]]
            df2 = df2[df2['MEA Concentration']==MEA_data[j]]
            df2 = df2.max()
       
            df_new = pd.concat([df_new,df2],axis=1)
            
    
    start = start+interval
    end = start+interval
    
df_new = df_new.T

wavenumber_data = df_new['Wavenumber'].unique()
df2 = pd.DataFrame()
for i in range(len(wavenumber_data)):
    df1 = pd.DataFrame(df_new[df_new['Wavenumber']==wavenumber_data[i]].iloc[:,2].values)
    df2 = pd.concat([df2,df1],axis=1)

df_new = df_new.iloc[:len(loading_data)*len(MEA_data),:].values

idx = [0,3,2,1]
df_new = df_new[:,idx]

X = df_new[:,1]
X = np.reshape(X,(-1,1))
Y = df_new[:,-1]
Y = np.reshape(Y,(-1,1))

df2 = df2.iloc[:,:].values
df_new = np.concatenate([X,df2,Y],axis=1)

X = df_new[:,:-1]
Y = df_new[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3) 

fs = SelectKBest(score_func=mutual_info_regression, k=3)
fs.fit(X_train, y_train)
X_fs = fs.transform(X)
X_fs = np.concatenate([df_new[:,0].reshape((-1,1)),X_fs],axis=1)

fig1 = plt.figure(dpi=400)
ax1 = fig1.add_axes([0,0,1,1])
ax1.bar(np.arange(1,len(col)+1,1),fs.scores_)
ax1.set_xticks(np.arange(1,len(col)+1,1),col,rotation=90)
ax1.set_ylabel('Mutual Information Regression Score')
ax1.set_xlabel('Features')
ax1.set_yticks(np.arange(0.0,1.2,0.2))
ax1.set_yticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'])
fig1.tight_layout()
score = fs.scores_