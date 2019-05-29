from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import zscore
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import os
import numpy as np
from sklearn import metrics
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

path = "./data/"
filename_read = os.path.join(path,"AirQualityUCI.csv")
filename_write = os.path.join(path,"air.csv")
df = pd.read_csv(filename_read,na_values=['NA','-200'])

# np.random.seed(42)
# df = df.reindex(np.random.permutation(df.index))
# df.reset_index(inplace=True, drop=True)

def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)

date = df['Date']
df.drop('Date',1,inplace=True)
time = df['Time']
df.drop('Time',1,inplace=True)
missing_median(df, 'CO(GT)')
missing_median(df, 'PT08.S1(CO)')
missing_median(df, 'NMHC(GT)')
missing_median(df, 'C6H6(GT)')
missing_median(df, 'PT08.S2(NMHC)')
missing_median(df, 'NOx(GT)')
missing_median(df, 'PT08.S3(NOx)')
missing_median(df, 'NO2(GT)')
missing_median(df, 'PT08.S4(NO2)')
missing_median(df, 'PT08.S5(O3)')
missing_median(df, 'T')
missing_median(df, 'RH')
missing_median(df, 'AH')

dataset=df.values
x=dataset[:,0:12]
print(x)
y=dataset[:,2]
print(y)
kf = KFold(5)

oos_y = []
oos_pred = []
fold = 0
for train, test in kf.split(x):
    fold+=1
    print("Fold #{}".format(fold))

x_train = x[train]
y_train = y[train]
x_test = x[test]
y_test = y[test]

model = Sequential()
model.add(Dense(128, input_dim=x.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=1,epochs=100)

pred = model.predict(x_test)
oos_y.append(y_test)
oos_pred.append(pred)
# Measure this fold's RMSE
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Fold score (RMSE): {}".format(score))

# Build the oos prediction list and calculate the error.
oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
score = np.sqrt(metrics.mean_squared_error(oos_pred,oos_y))
print("Final, out of sample score (RMSE): {}".format(score))

for i in range(10):
    print("{}. Co2: {}, predicted Co2: {}".format(i+1,y[i],pred[i]))

def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

# Plot the chart
chart_regression(pred.flatten(),y_test)
