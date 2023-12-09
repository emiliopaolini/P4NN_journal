import pandas as pd
import numpy as np
import nn

import sklearn.feature_selection as fs

from tensorflow import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler


# data from: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15?select=UNSW-NB15_LIST_EVENTS.csv

data = pd.read_csv("./data/cic_ids_2018.csv",header=0,sep=',',index_col=0)

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
data.drop_duplicates(inplace = True)

print(data["Label"].value_counts())

#data.drop(['Bwd IAT Mean','Bwd Pkts/s','Subflow Fwd Byts'],axis=1,inplace=True)


x_dataset = data.drop(['Label'], axis = 1)
X = x_dataset.to_numpy()
y_dataset = data['Label']
y = y_dataset.to_numpy()


print("X SHAPE: ",X.shape)
print("Y SHAPE: ",y.shape)

FEATURE_NUMBERS = 8
BITWIDTH = 4
CLASS_NUMBER = 4

selector = fs.SelectKBest(fs.f_classif, k=FEATURE_NUMBERS)
X = selector.fit_transform(X, y)

COLUMNS = data.columns
best_columns = selector.get_support(indices=True)
print("Best features: ",COLUMNS[best_columns])

print("X SHAPE: ",X.shape)
print("Y SHAPE: ",y.shape)

# Dataset normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


print("Y distribution: ",np.unique(y,return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,stratify=y)

print("Y train: ",np.unique(y_train,return_counts=True))
print("Y test: ",np.unique(y_test,return_counts=True))

exit()
# all features
X_train_tmp = data_tmp.iloc[:,[4,7,8,9,18,19,43,44]]
X_train_tmp.columns = ['IP_proto','s_bytes','d_bytes','s_TTL','swin','dwin','ct_src_ltm','ct_dst_ltm']

# stateless features
#X_train_tmp = data_tmp.iloc[:,[4,7,8,9,18,19]] 
#X_train_tmp.columns = ['IP_proto','s_bytes','d_bytes','s_TTL','swin','dwin']

# stateful features
#X_train_tmp = data_tmp.iloc[:,[43,44]]
#X_train_tmp.columns = ['ct_src_ltm','ct_dst_ltm']
 



y_train_tmp = data_tmp.iloc[:,48]
y_train_tmp.columns = ['class']

X_train_tmp = X_train_tmp[y_train_tmp==1]
y_train_tmp = y_train_tmp[y_train_tmp==1]
print("length of the y_train_tmp")
print(np.unique(y_train_tmp,return_counts=True))




# all features
X_train = data.iloc[:,[4,7,8,9,18,19,43,44]]
X_train.columns = ['IP_proto','s_bytes','d_bytes','s_TTL','swin','dwin','ct_src_ltm','ct_dst_ltm']

# stateless features
#X_train = data.iloc[:,[4,7,8,9,18,19]]
#X_train.columns = ['IP_proto','s_bytes','d_bytes','s_TTL','swin','dwin']
 
# stateful features
#X_train = data.iloc[:,[43,44]]
#X_train.columns = ['ct_src_ltm','ct_dst_ltm']



y_train = data.iloc[:,48]
y_train.columns = ['class']
print("Before merging")
print(np.unique(y_train,return_counts=True))
print("After merging")
X_train = pd.concat([X_train,X_train_tmp],axis=0)
y_train = pd.concat([y_train,y_train_tmp],axis=0)
print(np.unique(y_train,return_counts=True))


X_train = X_train.dropna()
X_train['IP_proto'] = X_train['IP_proto'].astype('category').cat.codes


scaler = MinMaxScaler()

X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25,stratify=y_train,shuffle=True)



print(np.unique(y_train,return_counts=True))
print(np.unique(y_test,return_counts=True))

# undersampling data
print("Data before balancing")
print(np.unique(y_train,return_counts=True))

undersampler = RandomUnderSampler(sampling_strategy='majority')

X_train,y_train = undersampler.fit_resample(X_train, y_train)


print("Training data after balancing")
print(np.unique(y_train,return_counts=True))
print("Test data")
print(np.unique(y_test,return_counts=True))





def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
  


input_1 = X_train['swin']
input_2 = X_train['s_TTL']

input_3 = X_train['dwin']
input_4 = X_train['IP_proto']

input_5 = X_train['ct_src_ltm']
input_6 = X_train['d_bytes'] #13

input_7 = X_train['ct_dst_ltm']
input_8 = X_train['s_bytes'] #13


input_1_val = X_test['swin']
input_2_val = X_test['s_TTL']

input_3_val = X_test['dwin']
input_4_val = X_test['IP_proto']

input_5_val = X_test['ct_src_ltm']
input_6_val = X_test['d_bytes']

input_7_val = X_test['ct_dst_ltm']
input_8_val = X_test['s_bytes']



opt = keras.optimizers.Adam(0.001)
model = nn.nn()

print(model.summary())

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy',f1_m,precision_m,recall_m])


checkpoint_filepath = '/tmp/model'+str(nn.bitwidth)+'all'

checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_f1_m',
                            verbose=1, save_best_only=True,save_weights_only=True, mode='max')

# all features
history = model.fit([input_1,input_2,input_3,input_4,input_5,input_6,input_7,input_8], y_train,
                    validation_data = ([input_1_val,input_2_val,input_3_val,input_4_val,input_5_val,input_6_val,input_7_val,input_8_val],y_test)
                    ,batch_size=256,epochs=50,shuffle=True,callbacks=[checkpoint])
# stateless
#history = model.fit([input_1,input_2,input_3,input_4,input_6,input_8], y_train,
#                    validation_data = ([input_1_val,input_2_val,input_3_val,input_4_val,input_6_val,input_8_val],y_test)
#                    ,batch_size=256,epochs=100,shuffle=True,callbacks=[checkpoint])
# stateful
#history = model.fit([input_5,input_7], y_train,
#                    validation_data = ([input_5_val,input_7_val],y_test)
#                    ,batch_size=256,epochs=100,shuffle=True,callbacks=[checkpoint])

model.load_weights(checkpoint_filepath)

model.save('models/unsw_stateless_'+str(nn.bitwidth)+'_bit.h5')

#print(model.evaluate([input_5_val,input_7_val],y_test,batch_size=256))
#print(model.evaluate([input_1_val,input_2_val,input_3_val,input_4_val,input_6_val,input_8_val],y_test,batch_size=256))
print(model.evaluate([input_1_val,input_2_val,input_3_val,input_4_val,input_5_val,input_6_val,input_7_val,input_8_val],y_test,batch_size=256))



selector = fs.SelectKBest(fs.f_classif, k=FEATURE_NUMBERS)
X = selector.fit_transform(X, y)

best_columns = selector.get_support(indices=True)

print("Best features: ",COLUMNS[best_columns])