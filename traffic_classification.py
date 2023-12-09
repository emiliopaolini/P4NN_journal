import new_nn as nn
import numpy as np
import pandas as pd
import sklearn.feature_selection as fs

from tensorflow import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

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

import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
import pandas as pd
import sklearn
import larq as lq

from collections import OrderedDict
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense,Conv2D,Conv2DTranspose,Flatten,Reshape,UpSampling2D,MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print_data = False

# Read all files with extension data from tmp folder 
filelist = glob.glob('./data/*.data')
preprocessed_flows = []
for file in filelist:
    with open(file, 'rb') as filehandle:
        # read the data as binary data stream
        print("Opening file {}".format(file))
        preprocessed_flows = preprocessed_flows + pickle.load(filehandle)

# define max flow length and time window
MAX_FLOW_LENGTH = 10
TIME_WINDOW = 10
X = []
y = []
for flow in preprocessed_flows:
    tuple = flow[0]
    flow_data = flow[1]
    label = flow_data['label']
    for key, fragment in flow_data.items():
        if key != 'label':
            X.append(fragment)
            y.append(label)

# print(np.unique(y,return_counts=True))
y = np.array(y)
#NORMALIZATION
protocols = ['arp','data','dns','ftp','http','icmp','ip','ssdp','ssl','telnet','tcp','udp']
powers_of_two = np.array([2**i for i in range(len(protocols))])
# feature list with min and max values
feature_list = OrderedDict([
    ('timestamp', [0,10]),
    ('packet_length',[0,1<<16]),
    ('highest_layer',[0,1<<32]),
    ('IP_flags',[0,1<<16]),
    ('protocols',[0,1<<len(protocols)]),
    ('TCP_length',[0,1<<16]),
    ('TCP_ack',[0,1<<32]),
    ('TCP_flags',[0,1<<16]),
    ('TCP_window_size',[0,1<<16]),
    ('UDP_length',[0,1<<16]),
    ('ICMP_type',[0,1<<8])]
)

# min/max values of features based on the nominal min/max values of the single features (as defined in the feature_list dict)
def static_min_max(time_window=TIME_WINDOW):
    feature_list['timestamp'][1] = time_window

    min_array = np.zeros(len(feature_list))
    max_array = np.zeros(len(feature_list))

    i=0
    for feature, value in feature_list.items():
        min_array[i] = value[0]
        max_array[i] = value[1]
        i+=1

    return min_array,max_array

mins, maxs = static_min_max()

def scale_linear_bycolumn(rawpoints, mins,maxs,high=1.0, low=0.0):
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)
def normalize_and_padding(X,mins,maxs,max_flow_len,padding=True):
    norm_X = []
    for sample in X:
        if sample.shape[0] > max_flow_len: # if the sample is bigger than expected, we cut the sample
            sample = sample[:max_flow_len,...]
        packet_nr = sample.shape[0] # number of packets in one sample
        norm_sample = scale_linear_bycolumn(sample, mins, maxs, high=1.0, low=0.0)
        np.nan_to_num(norm_sample, copy=False)  # remove NaN from the array
        if padding == True:
            norm_sample = np.pad(norm_sample, ((0, max_flow_len - packet_nr), (0, 0)), 'constant',constant_values=(0, 0))  # padding
        norm_X.append(norm_sample)
    return norm_X


# perform normalization and padding
X = normalize_and_padding(X,mins,maxs,MAX_FLOW_LENGTH,padding=True)

X = np.array(X)


FEATURE_NUMBERS = 8
BITWIDTH = 32





























exit()
df = pd.read_csv('data/Darknet.csv', header=0, sep=",", on_bad_lines="skip")
print(df)


df.rename(columns = {"Label" : "Type", "Label.1" : "Subtype"}, inplace = True)



df["Subtype"].loc[df["Subtype"] == "AUDIO-STREAMING"] = "Audio-Streaming"
df["Subtype"].loc[df["Subtype"] == "File-transfer"] = "File-Transfer"
df["Subtype"].loc[df["Subtype"] == "Video-streaming"] = "Video-Streaming"

df.drop('Type', inplace=True, axis=1)
df.drop('Flow ID', inplace=True, axis=1)
df.drop('Timestamp', inplace=True, axis=1)
df.drop('Src IP', inplace=True, axis=1)
df.drop('Dst IP', inplace=True, axis=1)
df.drop('Src Port', inplace=True, axis=1)
df.drop('Dst Port', inplace=True, axis=1)

sc = StandardScaler()

# Remove constant columns
#cols = [31,32,33,48,49,50,55,56,57,58,63,69,70,71,72]
#df.drop(df.columns[cols],axis=1,inplace=True)
COLUMNS = df.columns
print(COLUMNS)
print(df.info())

#print(df.loc[:,COLUMNS!='Subtype'].astype(np.float64))
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()


conversion_dict = {"Audio-Streaming":0,"Browsing":1, "Chat":2, "Email":3, "File-Transfer":4, "P2P":5, "VOIP":6, "Video-Streaming":7}
df['Subtype'] = df['Subtype'].replace(conversion_dict)


dataset = df.to_numpy()

X = dataset[:,:-1]
y = dataset[:,-1]

print("X before normalizing")
print(X)

X = sc.fit_transform(X)

print("X after normalizing")
print(X)

selector = fs.SelectKBest(fs.f_classif, k=FEATURE_NUMBERS)
X = selector.fit_transform(X, y)

best_columns = selector.get_support(indices=True)

print("Best features: ",COLUMNS[best_columns])

print(np.unique(y,return_counts=True))


if FEATURE_NUMBERS==8:
    # instantiate full model
    print("Instantiatin model with 7 LUTs")
    model = nn.nn(bitwidth=BITWIDTH,LUT=7,class_number=8)
if FEATURE_NUMBERS==6:
    # instantiate middle model
    print("Instantiatin model with 5 LUTs")
    model = nn.nn(bitwidth=BITWIDTH,LUT=5,class_number=8)
if FEATURE_NUMBERS==2:
    # instantiate small model
    print("Instantiatin model with 1 LUT")
    model = nn.nn(bitwidth=BITWIDTH,LUT=1,class_number=8)


opt = keras.optimizers.Adam(0.001)

#print(model.summary())
# TODO: NORMALIZE, SET XTRAIN AND XTEST

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy',f1_m,precision_m,recall_m])

checkpoint_filepath = '/tmp/model'+str(BITWIDTH)+'all'

checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='f1_m',
                            verbose=1, save_best_only=True,save_weights_only=True, mode='max')

# all features
history = model.fit([X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5],X[:,6],X[:,7]], y,batch_size=256,epochs=50,shuffle=True,callbacks=[checkpoint])

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