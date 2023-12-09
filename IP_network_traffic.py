# data from here https://www.kaggle.com/datasets/jsrojas/ip-network-traffic-flows-labeled-with-87-apps/code
import pandas as pd
# importing algorithms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# importing the required libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


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



sc = StandardScaler()

data = pd.read_csv("data/Dataset-Unicauca-Version2-87Atts.csv")



print('Number of Rows: {}'.format(data.shape[0]))
print('Number of Columns: {}'.format(data.shape[1]))


# Graph of Protocol Name vs Frequency
freq_protocol = data['ProtocolName'].value_counts()

# sns.histplot(freq_protocol.values())
application_name = []
frequency_count = []
for key, value in freq_protocol.items():
    application_name.append(key)
    frequency_count.append(value)
print(application_name)
print("Number of Unique Application Names: ", len(freq_protocol))    

# graph of top 5 application names
top_values = 5
plt.bar(application_name[:top_values], frequency_count[:top_values])
plt.xlabel("Application Name")
plt.ylabel("Frequency")


requiredProtocolName = ['GOOGLE','FACEBOOK','AMAZON','MICROSOFT']
'''
for key, value in freq_protocol.items():
    if (value >= 20000):
        requiredProtocolName.append(key)
'''
print(requiredProtocolName)

listofDataFrames = []
for protocol in requiredProtocolName:
    listofDataFrames.append(pd.DataFrame(data[data['ProtocolName'] == protocol].sample(n = 10000)))

sampledData = pd.concat(listofDataFrames)

# taking random rows and shuffling the dataframe
data = sampledData.sample(frac=1, random_state=1).reset_index()

# remove the rows that contains NULL values
data.dropna(inplace=True)
data.dropna(axis='columns')
data.reset_index(drop=True, inplace=True)

# remove columns which contains zeroes in the data
data = data.loc[:, (data != 0).any(axis=0)]


COLUMNS = data.columns
print('Shape after removing rows with NULL Values')
print('Number of Rows: {}'.format(data.shape[0]))
print('Number of Columns: {}'.format(data.shape[1]))



# converting the protocol name (target column) to required format (int)
# using LabelEncoder function from sklearn.preprocession library
encoder = LabelEncoder().fit(data['ProtocolName'])
data['ProtocolName'] = encoder.transform(data['ProtocolName'])
values = encoder.inverse_transform(data['ProtocolName'])
target_column = data['ProtocolName']

# mapping the encoded value
encoded_target_column = {}
for i in range(len(data['ProtocolName'])):
    encoded_target_column[data['ProtocolName'][i]] = values[i]

print(encoded_target_column)

dataset = data.drop(['Flow.ID','Source.IP','Label', 'Timestamp','Destination.IP', 'Source.Port', 'Destination.Port', 'Protocol'], axis=1)



x_dataset = dataset.drop(['ProtocolName'], axis = 1)
X = x_dataset.to_numpy()
y_dataset = dataset['ProtocolName']
y = y_dataset.to_numpy()
print("X SHAPE: ",X.shape)
print("Y SHAPE: ",y.shape)

FEATURE_NUMBERS = 8
BITWIDTH = 4
CLASS_NUMBER = 4

selector = fs.SelectKBest(fs.f_classif, k=FEATURE_NUMBERS)
X = selector.fit_transform(X, y)

best_columns = selector.get_support(indices=True)
print("Best features: ",COLUMNS[best_columns])

# normal dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



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


print("TRAIN",np.unique(y_train,return_counts=True))

print("TEST",np.unique(y_test,return_counts=True))

class_weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = np.array(y_train)
                                    )

dic_weights = dict(zip(np.unique(y_train), class_weights))
print("DICT",dic_weights)


opt = keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

checkpoint_filepath = '/tmp/model'+str(BITWIDTH)+'all'

checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy',
                            verbose=1, save_best_only=True,save_weights_only=True, mode='max')

print(y_train.shape)
print(X_train.shape)

if FEATURE_NUMBERS==8:
    # instantiate full model
    history = model.fit([X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3],X_train[:,4],X_train[:,5],X_train[:,6],X_train[:,7]], y_train,
                    batch_size=256,epochs=50,shuffle=True,class_weight= dic_weights,callbacks=[checkpoint],
                    validation_data = ([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3],X_test[:,4],X_test[:,5],X_test[:,6],X_test[:,7]],y_test))

    model.load_weights(checkpoint_filepath)
    

    print(model.evaluate([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3],X_test[:,4],X_test[:,5],X_test[:,6],X_test[:,7]],y_test,batch_size=256))

if FEATURE_NUMBERS==6:
    # instantiate middle model
    history = model.fit([X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3],X_train[:,4],X_train[:,5]], y_train,
                    batch_size=256,epochs=50,shuffle=True,class_weight= dic_weights,callbacks=[checkpoint],
                    validation_data = ([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3],X_test[:,4],X_test[:,5]],y_test))

    model.load_weights(checkpoint_filepath)
    

    print(model.evaluate([X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3],X_test[:,4],X_test[:,5]],y_test,batch_size=256))
if FEATURE_NUMBERS==2:
    history = model.fit([X_train[:,0],X_train[:,1]], y_train,
                    batch_size=256,epochs=50,shuffle=True,class_weight= dic_weights,callbacks=[checkpoint],
                    validation_data = ([X_test[:,0],X_test[:,1]],y_test))

    model.load_weights(checkpoint_filepath)
    

    print(model.evaluate([X_test[:,0],X_test[:,1]],y_test,batch_size=256))

model.save('models/IP_network_traffic_bit_'+str(BITWIDTH)+'_feature_'+str(FEATURE_NUMBERS)+'.h5')
exit()

