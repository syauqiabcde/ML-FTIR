#Importing Libraries
import pandas as pd
import numpy as np
import pickle
from scipy.optimize import root,minimize
from google.colab import drive
from IPython.display import clear_output
drive.mount('/content/drive')

path ='/content/drive/MyDrive/Colab Notebooks/MEA and CO2 Loading Prediction/operation.xlsx'

df = pd.read_excel(path)
col = ['Wavenumber','Absorbance']
df = pd.DataFrame(df.iloc[:,:2].values,columns=col)

def MEA_prediction(CO2):
    CO2 = np.array([CO2]).reshape(-1,1)
    start = [952, 1028]
    interval = 5
    df3 = pd.DataFrame()
    for i in range(len(start)):
        start_1 = start[i]-interval
        end = start[i]+interval
        df1 = df[(df['Wavenumber'].between(start_1,end))]
        df1 = df1.sort_values(by=['Wavenumber'])
        df1 = df1.mean()
        df3 = pd.concat([df3,df1],axis=1)

    df3 = df3.iloc[:,:].values
    df3 = np.delete(df3,0,0)
    X = np.concatenate([df3, CO2],axis=1)

    path_MEA = '/content/drive/MyDrive/Colab Notebooks/MEA and CO2 Loading Prediction/XGB_MEA_Concentration'
    MEA_model = pickle.load(open(path_MEA, 'rb'))
    MEA = MEA_model.predict(X)
    return MEA

def CO2_prediction(MEA):
    MEA = np.array([MEA]).reshape(-1,1)
    start = [1488, 1580]
    interval = 5
    df3 = pd.DataFrame()
    for i in range(len(start)):
        start_1 = start[i]-interval
        end = start[i]+interval
        df1 = df[(df['Wavenumber'].between(start_1,end))]
        df1 = df1.mean()
        df3 = pd.concat([df3,df1],axis=1)

    df3 = df3.iloc[:,:].values
    df3 = np.delete(df3,0,0)
    X = np.concatenate([df3, MEA],axis=1)

    path_CO2 = '/content/drive/MyDrive/Colab Notebooks/MEA and CO2 Loading Prediction/ADB_CO2_Concentration_v1'
    CO2_model = pickle.load(open(path_CO2, 'rb'))
    CO2 = CO2_model.predict(X)
    CO2 = max(0,CO2)
    return CO2

def func(CO2):
    MEA = MEA_prediction(CO2)
    loading = CO2_prediction(MEA)
    error = abs(CO2-loading)*100
    return error

y = minimize(func,0,method='Powell')
loading = y.x
MEA = MEA_prediction(loading)

clear_output()
print(f'The CO2 Loading is {loading[0]:.4f} mol/mol')
print(f'The MEA Concentration is {MEA[0]*100:.2f} %')
