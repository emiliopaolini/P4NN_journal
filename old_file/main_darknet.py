import new_nn as nn
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.feature_selection as fs
import tensorflow_addons as tfa 

from sklearn.utils import class_weight
from tensorflow import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN
from sklearn.preprocessing import StandardScaler,MinMaxScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import  OneHotEncoder
from sklearn.decomposition import PCA

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
  



FEATURE_NUMBERS = 10
BITWIDTH = 16
CLASS_NUMBER = 8

df = pd.read_csv('data/Darknet.csv', header=0, sep=",", on_bad_lines="skip")
print(df)


df.rename(columns = {"Label" : "Type", "Label.1" : "Subtype"}, inplace = True)


df["Subtype"].loc[df["Subtype"] == "AUDIO-STREAMING"] = "Audio-Streaming"
df["Subtype"].loc[df["Subtype"] == "File-transfer"] = "File-Transfer"
df["Subtype"].loc[df["Subtype"] == "Video-streaming"] = "Video-Streaming"

df.drop('Type', inplace=True, axis=1)
df.drop('Flow ID', inplace=True, axis=1)
df.drop('Flow Bytes/s', inplace=True, axis=1)
df.drop('Flow Packets/s', inplace=True, axis=1)
df.drop('Timestamp', inplace=True, axis=1)
#df.drop('Src IP', inplace=True, axis=1)
#df.drop('Dst IP', inplace=True, axis=1)
#df.drop('Src Port', inplace=True, axis=1)
#df.drop('Dst Port', inplace=True, axis=1)

le = LabelEncoder()
df['Src IP'] = le.fit_transform(df['Src IP'])
df['Dst IP'] = le.fit_transform(df['Dst IP'])

# Remove constant columns
#cols = [31,32,33,48,49,50,55,56,57,58,63,69,70,71,72]
#df.drop(df.columns[cols],axis=1,inplace=True)
COLUMNS = df.columns
print(COLUMNS)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

#df.iloc[:,0:-1] = df.iloc[:,0:-1].apply(lambda x: (x-x.mean())/ x.std(), axis=0)

scaler = MinMaxScaler()
df.iloc[:,0:-1] = scaler.fit_transform(df.iloc[:,0:-1].to_numpy())

print(df)
#print(df.loc[:,COLUMNS!='Subtype'].astype(np.float64))



conversion_dict = {"Audio-Streaming":0,"Browsing":1, "Chat":2, "Email":3, "File-Transfer":4, "P2P":5, "VOIP":6, "Video-Streaming":7}

df['Subtype'] = df['Subtype'].replace(conversion_dict)


dataset = df.to_numpy()

X = dataset[:,:-1]
y = dataset[:,-1].astype(float)

if FEATURE_NUMBERS==10:
    # instantiate full model
    print("Instantiatin model with 7 LUTs")
    model = nn.nn(bitwidth=BITWIDTH,LUT=10,class_number=CLASS_NUMBER)
    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer= opt, metrics=['accuracy'],loss='sparse_categorical_crossentropy')


    f1s = []
    precisions = []
    recalls = []
    history = model.fit(X, y, batch_size = 128, shuffle=True,epochs = 10, verbose = True)


'''
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
'''

selector = fs.SelectKBest(fs.f_classif, k=FEATURE_NUMBERS)
X = selector.fit_transform(X, y)

best_columns = selector.get_support(indices=True)

print("Best features: ",COLUMNS[best_columns])

#print(df.info())

#print(df[COLUMNS[best_columns]].agg(['min', 'max']))

X = df[COLUMNS[best_columns]].to_numpy()
print(np.unique(y,return_counts=True))





if FEATURE_NUMBERS==8:
    # instantiate full model
    print("Instantiatin model with 7 LUTs")
    model = nn.nn(bitwidth=BITWIDTH,LUT=7,class_number=CLASS_NUMBER)
if FEATURE_NUMBERS==6:
    # instantiate middle model
    print("Instantiatin model with 5 LUTs")
    model = nn.nn(bitwidth=BITWIDTH,LUT=5,class_number=CLASS_NUMBER)
if FEATURE_NUMBERS==2:
    # instantiate small model
    print("Instantiatin model with 1 LUT")
    model = nn.nn(bitwidth=BITWIDTH,LUT=1,class_number=CLASS_NUMBER)




#print(model.summary())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,shuffle=True)

'''
oversampler = RandomOverSampler(sampling_strategy='all')
#oversampler = ADASYN()
X_train,y_train = oversampler.fit_resample(X_train, y_train)
'''

class_weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = np.array(y_train)
                                    )

dic_weights = dict(zip(np.unique(y_train), class_weights))
print("DICT",dic_weights)


opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

checkpoint_filepath = '/tmp/model'+str(BITWIDTH)+'all'

checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='f1_score',
                            verbose=1, save_best_only=True,save_weights_only=True, mode='max')


'''
encoder = OneHotEncoder(sparse_output=False)
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(y_train.reshape(-1, 1))
y_train = encoder.transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))
'''

# all features
history = model.fit([X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3],X_train[:,4],X_train[:,5],X_train[:,6],X_train[:,7]], y_train,
                    batch_size=256,epochs=50,shuffle=True,class_weight= dic_weights,callbacks=[checkpoint],
                    validation_data = ([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3],X_test[:,4],X_test[:,5],X_test[:,6],X_test[:,7]],y_test))

exit()


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