import pandas as pd
import numpy as np

df = pd.read_csv('Dataset.csv')
interval = 5
# wn_MEA = [671, 1028]
wn_MEA = [952, 1328]
wn_CO2 = [1489, 1561]

CO2_values = df['CO2 Loading'].unique() 
MEA_values = df['MEA Concentration'].unique() 

df_CO2 = pd.DataFrame()
df_MEA = pd.DataFrame()
# df_2 = pd.DataFrame()
# for i in range(len(CO2_values)):
#     CO2 = CO2_values[i]
#     for j in range(len(MEA_values)):
#         MEA = MEA_values[j]
#         df_1 = df[(df['CO2 Loading']==CO2) & (df['MEA Concentration']==MEA)]
#         min_abs = df_1['Absorbance'].min()
#         df_1.iloc[:,1] = df_1.iloc[:,1] - min_abs
#         df_2 = pd.concat([df_2,df_1])
        
# df = df_2

for i in range(len(wn_CO2)):
    start = wn_CO2[i]-interval
    end = wn_CO2[i]+interval    
    df1 = df[(df['Wavenumber'].between(start,end))]
    df_CO2 = pd.concat([df_CO2,df1])

for i in range(len(wn_MEA)):
    start = wn_MEA[i]-interval
    end = wn_MEA[i]+interval      
    df1 = df[(df['Wavenumber'].between(start,end))]
    df_MEA = pd.concat([df_MEA,df1])
    


CO2_array = np.array([100, 100, 100, 100]).reshape(1,-1)
MEA_array = np.array([100, 100, 100, 100]).reshape(1,-1)

for i in range(len(CO2_values)):
    CO2 = CO2_values[i]
    for j in range(len(MEA_values)):
        MEA = MEA_values[j]
        df1 = df_CO2[(df_CO2['CO2 Loading']==CO2) & (df_CO2['MEA Concentration']==MEA)]
        df_low = df1[df1['Wavenumber'] <= wn_CO2[0]+interval].to_numpy()
        df_high = df1[df1['Wavenumber'] > wn_CO2[0]+interval].to_numpy()
        mean_low = np.mean(df_low, axis=0).reshape(1,-1)
        mean_high = np.mean(df_high, axis=0).reshape(1,-1)
        new_arr = np.array([mean_low[0,1], mean_high[0,1], mean_high[0,2], mean_high[0,3]]).reshape(1,-1)
        CO2_array = np.concatenate((CO2_array, new_arr))
        
nan_mask = np.isnan(CO2_array).any(axis=1)
CO2_array = CO2_array[~nan_mask][1:,:]

for i in range(len(CO2_values)):
    CO2 = CO2_values[i]
    for j in range(len(MEA_values)):
        MEA = MEA_values[j]
        df1 = df_MEA[(df_MEA['CO2 Loading']==CO2) & (df_MEA['MEA Concentration']==MEA)]
        df_low = df1[df1['Wavenumber'] <= wn_MEA[0]+5].to_numpy()
        df_high = df1[df1['Wavenumber'] > wn_MEA[0]+5].to_numpy()
        mean_low = np.mean(df_low, axis=0).reshape(1,-1)
        mean_high = np.mean(df_high, axis=0).reshape(1,-1)
        new_arr = np.array([mean_low[0,1], mean_high[0,1], mean_high[0,2], mean_high[0,3]]).reshape(1,-1)
        MEA_array = np.concatenate((MEA_array, new_arr))
        
nan_mask = np.isnan(MEA_array).any(axis=1)
MEA_array = MEA_array[~nan_mask][1:,:][:,[0,1,3,2]]

#%%

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoLars
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, DotProduct, ExpSineSquared
from sklearn.linear_model import HuberRegressor, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from tensorflow.keras.layers import Input,Dense,BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import layers, models
from sklearn.svm import SVR
from xgboost import XGBRegressor

# X = MEA_array[:,:-1]
# Y = MEA_array[:,-1]
# CO2_array = np.delete(CO2_array,[12,32],0)
X = CO2_array[:,:-1]
Y = CO2_array[:,-1]
# scale = MinMaxScaler(feature_range=(-1,1))
# scale = StandardScaler()

# X = scale.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
kernel = DotProduct()
# model =  GaussianProcessRegressor(kernel=kernel)
# model = LassoLars()
# model = SVR(kernel='rbf')
# model = PLSRegression()
# model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1, positive=False)
# model = BayesianRidge()
# model = ExtraTreesRegressor()
# model = HuberRegressor()
model = AdaBoostRegressor()
# model = XGBRegressor()
# model = RandomForestRegressor(n_estimators=1000)
# model.fit(X_train, y_train)
model.fit(X,Y)

# accuracies = cross_val_score(estimator=model, X=X, y=Y, cv=5, scoring='r2')
# mean = accuracies.mean()
# std  = accuracies.std()

# Y_pred_train = model.predict(X_train)
# Y_pred = model.predict(X)
# Y_pred_test = model.predict(X_test)
# R2_test = r2_score(y_test, Y_pred_test)
# R2_train = r2_score(y_train, Y_pred_train)
# R2 = r2_score(Y, Y_pred)
# print('mean R2 CV: ', mean)
# print('Std R2 CV: ', std)
# print('R2 train: ', R2_train)
# print('R2_test: ', R2_test)

Y_pred = model.predict(X)
R2 = r2_score(Y, Y_pred)
RMSE = mean_squared_error(Y, Y_pred, squared=False)
print('R2 Total: ',R2)
print('RMSE Total: ',RMSE)

#%%
import pickle
pickle.dump(model, open('ADB_CO2_Concentration_v1', "wb"))

# import joblib
# scaler_filename = "MEA scaler.save"
# joblib.dump(scale, scaler_filename) 

#%%
