# Import packages
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dropout, Conv2D, Flatten,Dense,  MaxPooling2D, GlobalAveragePooling2D

# Control the memory of GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess = tf.compat.v1.Session(config=config)

# Normalization the data
def normalization(data):
    data_range = np.max(data) - np.min(data)
    if 0 == data_range:
        return data
    else:
        return (data - np.min(data)) / data_range

# Read the samples( Selecte the channels of interested)
N = 2;
# N : You can choose 2,4,6,8
# N = 2 -> usecols = [10,45]
# N = 4 -> usecols = [10,12,45,49]
# N = 6 -> usecols = [10,12,19,45,49,56]
# N = 8 -> usecols = [10,12,13,19,45,49,50,56]

data = np.empty([1,512,N,1], dtype = float)
label = []
for info in os.listdir('eeg_data/motor_left'):
    domain = os.path.abspath(r'eeg_data/motor_left')
    info = os.path.join(domain, info)
    data1 = pd.read_csv(info, header=0, nrows=512, usecols=[10,12,19,45,49,56])
    data1 = np.array(data1)
    data1 = normalization(data1)
    data1 = np.expand_dims(data1,axis=0)
    data1 = np.expand_dims(data1, axis=-1)
    data=np.append(data,data1,axis=0)
    label.append(0)
for info in os.listdir('eeg_data/motor_right'):
    domain = os.path.abspath(r'eeg_data/motor_right')
    info = os.path.join(domain,info)
    data2 = pd.read_csv(info,header=0,nrows=512,usecols=[10,12,19,45,49,56])
    data2 = np.array(data2)
    data2 = normalization(data2)
    data2=np.expand_dims(data2,axis=0)
    data2=np.expand_dims(data2,axis=-1)
    data=np.append(data,data2,axis=0)
    label.append(1)

# Transform label into one hot code
X=np.array(data[1:])
label=np.array(label)
Y=to_categorical(label)

# Divide the data set
train_x,test_x,train_y,test_y=train_test_split(X, Y, test_size=0.2,random_state=32,shuffle=True)
random_state= 32

#Build the model
model1=Sequential()
model1.add(Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),activation='tanh',padding="same",input_shape=(512,N,1)))
model1.add(Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),activation='tanh',padding="same"))
model1.add(Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),activation='tanh',padding="same"))
model1.add(MaxPooling2D(pool_size=(6, 1),padding="same"))
model1.add(Dropout(0.5))
model1.add(Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),activation='tanh',padding="same"))
model1.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),activation='tanh',padding="same"))
model1.add(MaxPooling2D(pool_size=(6,1),padding="same"))
model1.add(Dropout(0.5))
model1.add(GlobalAveragePooling2D())
model1.add(Flatten())
model1.add(Dense(2,activation='softmax'))
model1.summary()
model1.compile(optimizer=Adam(lr=0.004), loss='binary_crossentropy', metrics=['acc'])
history = model1.fit(train_x, train_y, validation_split=0.25, batch_size=64, epochs =250 )

# Calculate confusion_matrix operation
predict = model1.predict(test_x)
real_left =0 ;left_true_positive=0;left_true_negative=0;real_right =0 ;right_true_positive=0;right_true_negative=0
for i in range (0,1664):
    if(test_y[i,0]==1):
        real_left+=1
        if(predict[i,0]>predict[i,1]):
            left_true_positive+=1
        else:
            left_true_negative+=1
    if(test_y[i,1]==1):
        real_right+=1
        if(predict[i,0]<predict[i,1]):
            right_true_positive+=1
        else:
            right_true_negative+=1
print("the test num is:",real_left + real_right)
print("the real left num is:",real_left)
print("the left_true_positive num is:",left_true_positive)
print("the left_true_negative num is:",left_true_negative)
print("Accuracy of motor_left:",left_true_positive / real_left)
print("the real right num is:",real_right)
print("the right_true_positive num is:",right_true_positive)
print("the right_true_negative num is:",right_true_negative)
print("Accuracy of motor_right:",right_true_positive / real_right)

# Save the model
import time
time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
str = '2D_2CHAN' # Can selecte 2D_2CHAN ,2D_4CHAN, 2D_6CHAN, 2D_8CHAN
model1.save('{}_{}_model_{}.h5'.format(str,random_state,time_stamp))
# model1.save_weights('{}_{}_weights_{}.h5'.format(str,random_state,time_stamp))

# Calculate and print the loss and acc on our test dataset
test_loss,test_acc=model1.evaluate(test_x,test_y)
print("test_loss= ",test_loss)
print("test_acc= ",test_acc)

#plot the loss of train set and validate set
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right')
plt.show()

#plot the acc of train set and validate set
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model train vs validation acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right')
plt.show()
