import pandas as pd
import numpy as np
import sklearn.feature_selection as fs
import new_nn as nn
import tensorflow_addons as tfa 


from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from tensorflow import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN
from sklearn.preprocessing import StandardScaler,MinMaxScaler, LabelEncoder,RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import  OneHotEncoder
from sklearn.decomposition import PCA

FEATURE_NUMBERS = 8
BITWIDTH = 8
CLASS_NUMBER = 1

data_1 = loadarff('data/TimeBasedFeatures-15s-TOR-NonTOR.arff')
df = pd.DataFrame(data_1[0])


data_2 = loadarff('data/TimeBasedFeatures-30s-TORNonTOR.arff')
df = pd.concat([df,pd.DataFrame(data_2[0])])

#data_3 = loadarff('data/TimeBasedFeatures-60s-TOR-NonTOR.arff')
#df = pd.concat([df,pd.DataFrame(data_3[0])])

data_4 = loadarff('data/TimeBasedFeatures-120s-TOR-NonTOR.arff')
df = pd.concat([df,pd.DataFrame(data_4[0])])

print(df.head())

df['class1'] = df['class1'].astype(str)
#df = df[~df['class1'].str.contains("VPN")]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

scaler = RobustScaler()
df.iloc[:,0:-1] = scaler.fit_transform(df.iloc[:,0:-1].to_numpy())


print(df)

print(df["class1"].unique())

'''
conversion_dict = {'CHAT':0,
                   'VPN-CHAT':1,
                   'FT':0,
                   'VPN-FT':1,
                   'MAIL':0,
                   'VPN-MAIL':1,
                   'STREAMING':0,
                   'VPN-STREAMING':1,
                   'VOIP':0,
                   'VPN-VOIP':1,
                   'P2P':0,
                   'VPN-P2P':1,
                   'BROWSING':0,
                   'VPN-BROWSING':1
                   }
'''
conversion_dict = {'NONTOR':0,
                   'TOR':1,
                   }


df['class1'] = df['class1'].replace(conversion_dict)

COLUMNS = df.columns

dataset = df.to_numpy()

X = dataset[:,:-1]
y = dataset[:,-1].astype(float)

print(np.unique(y,return_counts=True))

selector = fs.SelectKBest(fs.f_classif, k=FEATURE_NUMBERS)
X = selector.fit_transform(X, y)

best_columns = selector.get_support(indices=True)

print("Best features: ",COLUMNS[best_columns])
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



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,shuffle=True)

print("TRAIN",np.unique(y_train,return_counts=True))

print("TEST",np.unique(y_test,return_counts=True))


#oversampler = RandomOverSampler(sampling_strategy='all')
#oversampler = ADASYN()
oversampler = SMOTE()
X_train,y_train = oversampler.fit_resample(X_train, y_train)


class_weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = np.array(y_train)
                                    )

dic_weights = dict(zip(np.unique(y_train), class_weights))
print("DICT",dic_weights)


opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy',tfa.metrics.F1Score(num_classes=CLASS_NUMBER,average='weighted')])

checkpoint_filepath = '/tmp/model'+str(BITWIDTH)+'all'

checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_f1_score',
                            verbose=1, save_best_only=True,save_weights_only=True, mode='max')


if FEATURE_NUMBERS==8:
    # instantiate full model
    history = model.fit([X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3],X_train[:,4],X_train[:,5],X_train[:,6],X_train[:,7]], y_train,
                    batch_size=256,epochs=50,shuffle=True,class_weight= dic_weights,callbacks=[checkpoint],
                    validation_data = ([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3],X_test[:,4],X_test[:,5],X_test[:,6],X_test[:,7]],y_test))

    model.load_weights(checkpoint_filepath)
    

    print(model.evaluate([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3],X_test[:,4],X_test[:,5],X_test[:,6],X_test[:,7]],y_test,batch_size=256))

if FEATURE_NUMBERS==6:
    # instantiate middle model
    print("Instantiatin model with 5 LUTs")
    model = nn.nn(bitwidth=BITWIDTH,LUT=5,class_number=CLASS_NUMBER)
if FEATURE_NUMBERS==2:
    # instantiate small model
    print("Instantiatin model with 1 LUT")
    model = nn.nn(bitwidth=BITWIDTH,LUT=1,class_number=CLASS_NUMBER)

model.save('models/tor_nontor_bit_'+str(BITWIDTH)+'_feature_'+str(FEATURE_NUMBERS)+'.h5')
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