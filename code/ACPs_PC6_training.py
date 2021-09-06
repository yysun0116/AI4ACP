import numpy as np
import pandas as pd
from Protein_Encoding import PC_6

# import data
ACP_data = PC_6('./data/new/train_test/4db_all_cdhit99_2124.fasta', length=50)
non_ACP_data = PC_6('./data/new/train_test/neg_all_2124.fasta', length=50)
# turn the list into array
ACP_array= np.array(list(ACP_data.values()))
non_ACP_array = np.array(list(non_ACP_data.values()))

features = np.concatenate((non_ACP_array,ACP_array),axis=0)
labels = np.hstack((np.repeat(0, len(non_ACP_data)),np.repeat(1, len(ACP_data))))

# shuffle numpy array
idxs = np.arange(features.shape[0])
np.random.shuffle(idxs)
x = features[idxs]
y = labels[idxs]


# training model
import tensorflow.keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input,Flatten,Masking,BatchNormalization
from tensorflow.keras.layers import LSTM,Conv1D, MaxPool1D
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import L1, L2
import os

# model architechure
def t_m(train_data, train_label, model_name, path = None):
    input_ = Input(shape=(50,6))
    cnn = Conv1D(filters = 64, kernel_size = 20, strides = 1,activation = 'relu', padding="same")(input_)
    norm = BatchNormalization()(cnn)
    MaxPool = MaxPool1D(pool_size=2, strides=1, padding='same')(norm)
    drop = Dropout(0.25)(MaxPool)
    
    cnn2 = Conv1D(filters = 32, kernel_size = 20, strides = 1,activation = 'relu', padding="same")(drop)
    norm2 = BatchNormalization()(cnn2)
    MaxPool2 = MaxPool1D(pool_size=2, strides=1, padding='same')(norm2)
    drop = Dropout(0.25)(MaxPool2)
    
    cnn3 = Conv1D(filters = 8, kernel_size = 20, strides = 1,activation = 'relu', padding="same")(drop)
    norm3 = BatchNormalization()(cnn3)
    MaxPool3 = MaxPool1D(pool_size=2, strides=1, padding='same')(norm3)
    drop = Dropout(0.25)(MaxPool3)

    Flat = Flatten()(drop)
    Den = Dense(128, activation = "relu")(Flat)
    drop = Dropout(0.5)(Den)
    result = Dense(1, activation = "sigmoid" ,kernel_regularizer= L2(0.01),activity_regularizer= L2(0.01))(drop)
    model = Model(inputs=input_,outputs=result)

    model.compile(optimizer=optimizers.Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    e_s = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=100,
                                          verbose=0, mode='min')
    best_weights_filepath = path+'/%s_best_weights.h5'%model_name
    saveBestModel = tensorflow.keras.callbacks.ModelCheckpoint(best_weights_filepath, 
                                                        monitor='val_loss', 
                                                        verbose=1, 
                                                        save_best_only=True, 
                                                        mode='auto')
    CSVLogger = tensorflow.keras.callbacks.CSVLogger(path+"/%s_csvLogger.csv"%model_name,separator=',', append=False)

    t_m=model.fit(train_data,train_label,shuffle=True,validation_split=0.1, 
                    epochs=500, batch_size=int(0.1*len(train_data)),callbacks=[saveBestModel,CSVLogger])
    return model,t_m


# training model
model, t_m = t_m(x,y,'PC_6_model_ACP4db_final', path='/home/yysun0116/ACPs/PC6/')



# show the model training process
from sklearn.metrics import accuracy_score,accuracy_score,f1_score,matthews_corrcoef,confusion_matrix,roc_curve,auc
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(t_m ,'accuracy','val_accuracy')
show_train_history(t_m ,'loss','val_loss')