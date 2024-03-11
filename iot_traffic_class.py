import pandas as pd
import numpy as np
import tensorflow as tf
import new_nn_traffic as nn
import tensorflow_addons as tfa
import single_lut_nn
from keras.utils import to_categorical
import sklearn.feature_selection as fs
from sklearn.utils import class_weight
from tensorflow import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder

#https://sites.google.com/view/iot-network-intrusion-dataset/home?pli=1

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
    
    return macro_f1


def convert_columns(data):
    column_to_convert = ['proto']
    for column in column_to_convert:
        data[column] = pd.Categorical(data[column])
        data[column] = data[column].cat.codes


data = pd.read_csv("./data/IoT Network Intrusion Dataset.csv",sep=',',header=0)

#data = pd.concat([data,data_2])


#print(data['Sub_Cat'].value_counts())


data.drop(['Flow_ID','Src_Port','Src_IP','Dst_IP','Dst_Port','Sub_Cat','Label','Timestamp'],axis=1,inplace=True)


data = data[data.columns.drop(list(data.filter(regex='Mean')))]
data = data[data.columns.drop(list(data.filter(regex='Std')))]

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
data.drop_duplicates(inplace = True)
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    #print(data['Label'].value_counts())
    print(data['Cat'].value_counts())
    #print(data['Sub_Cat'].value_counts())
    
#convert_columns(data)

print(data.head())


COLUMNS = data.columns

#print(COLUMNS)



labels_to_keep = ["Normal","DoS","Scan","MITM ARP Spoofing"]
conversion_dict = {"Normal":0,"DoS":2,"Scan":3,"MITM ARP Spoofing":4}

data = data.loc[data['Cat'].isin(labels_to_keep)]
data['Cat'] = data['Cat'].replace(conversion_dict)


encoder = LabelEncoder().fit(data['Cat'])
data['Cat'] = encoder.transform(data['Cat'])
print(data['Cat'])


x_dataset = data.drop(['Cat'], axis = 1)
X = x_dataset.to_numpy()
y_dataset = data['Cat']
y = y_dataset.to_numpy()

print("X SHAPE: ",X.shape)
print("Y SHAPE: ",y.shape)

FEATURE_NUMBERS = 8
BITWIDTH = 6
CLASS_NUMBER = 4



selector = fs.SelectKBest(fs.f_classif, k=FEATURE_NUMBERS)
X = selector.fit_transform(X, y)

#FEATURE_NUMBERS = -1

best_columns = selector.get_support(indices=True)
print("Best features: ",COLUMNS[best_columns])

# Dataset normalization
scaler = MinMaxScaler((0,1))
X = scaler.fit_transform(X)

#print(X)

print("Y distribution: ",np.unique(y,return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print("Data before balancing")
print(np.unique(y_train,return_counts=True))

print("Y train: ",np.unique(y_train,return_counts=True))
print("Y test: ",np.unique(y_test,return_counts=True))
'''
undersampler = RandomUnderSampler(sampling_strategy='all')
X_train,y_train = undersampler.fit_resample(X_train, y_train)
'''
print("Data after balancing")
print(np.unique(y_train,return_counts=True))

if FEATURE_NUMBERS == -1:
    model = single_lut_nn.nn(bitwidth=BITWIDTH,LUT=1,class_number=CLASS_NUMBER)

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
 
class_weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = np.array(y_train)
                                    )


dic_weights = dict(zip(np.unique(y_train), class_weights))
print("DICT",dic_weights)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

opt = keras.optimizers.Adam(0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy',f1_metric])

checkpoint_filepath = '/tmp/model'+str(BITWIDTH)+'all'

checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_f1_metric',
                            verbose=1, save_best_only=True,save_weights_only=True, mode='max')


if FEATURE_NUMBERS==-1:
    history = model.fit(X_train, y_train,
                    batch_size=256,epochs=50,shuffle=True,class_weight= dic_weights,callbacks=[checkpoint],
                    validation_data = (X_test,y_test))

    model.load_weights(checkpoint_filepath)
    

    print(model.evaluate(X_test,y_test,batch_size=256))

if FEATURE_NUMBERS==8:
    print("FEATURE NUMBERS 8")
    # instantiate full model

    history = model.fit([X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3],X_train[:,4],X_train[:,5],X_train[:,6],X_train[:,7]], y_train,
                    batch_size=256,epochs=100,shuffle=True,class_weight= dic_weights,callbacks=[checkpoint],
                    validation_data = ([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3],X_test[:,4],X_test[:,5],X_test[:,6],X_test[:,7]],y_test))

    model.load_weights(checkpoint_filepath)
    

    print(model.evaluate([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3],X_test[:,4],X_test[:,5],X_test[:,6],X_test[:,7]],y_test,batch_size=256))

if FEATURE_NUMBERS==6:
    # instantiate middle model
    history = model.fit([X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3],X_train[:,4],X_train[:,5]], y_train,
                    batch_size=128,epochs=30,shuffle=True,class_weight= dic_weights,callbacks=[checkpoint],
                    validation_data = ([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3],X_test[:,4],X_test[:,5]],y_test))

    model.load_weights(checkpoint_filepath)
    

    print(model.evaluate([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3],X_test[:,4],X_test[:,5]],y_test,batch_size=256))
if FEATURE_NUMBERS==2:
    history = model.fit([X_train[:,0],X_train[:,1]], y_train,
                    batch_size=256,epochs=30,shuffle=True,class_weight= dic_weights,callbacks=[checkpoint],
                    validation_data = ([X_test[:,0],X_test[:,1]],y_test))

    model.load_weights(checkpoint_filepath)
    

    print(model.evaluate([X_test[:,0],X_test[:,1]],y_test,batch_size=256))

model.save('models/IP_network_traffic_bit_'+str(BITWIDTH)+'_feature_'+str(FEATURE_NUMBERS)+'.h5')
exit()