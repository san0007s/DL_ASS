from tensorflow.keras.utils import to_categorical
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models,optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv",header=None)
pd.DataFrame(data)

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

standard = StandardScaler()
x_scaled = standard.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.3,random_state=42)

print(x_train.shape[1])

input_dim = x_train.shape[1]

encoder_input = tf.keras.Input(shape=input_dim)
encoded = Dense(128,activation='relu')(encoder_input)
encoded = Dropout(0.2)(encoded)
encoded = Dense(64,activation='relu')(encoded)
encoded = Dropout(0.2)(encoded)
encoded = Dense(32,activation='relu')(encoded)
latent = Dense(16,activation='relu')(encoded)

decoded = Dense(16,activation='relu')(latent)
decoded = Dense(32,activation='relu')(decoded)
decoded = Dropout(0.2)(decoded)
decoded = Dense(64,activation='relu')(decoded)
decoded = Dense(128,activation='relu')(decoded)
decoder_output = Dense(input_dim,activation='sigmoid')(decoded)

autoencoder = models.Model(inputs =encoder_input,outputs= decoder_output)


autoencoder.compile(optimizer = optimizers.Adam(learning_rate=0.01),
                    loss = 'mse',
                    metrics=['mae'])

history = autoencoder.fit(x_train,x_train,epochs=50,batch_size=64,validation_data=(x_test,x_test))

reconstructions =autoencoder.predict(x_test)
reconstructions_error = np.mean(np.square(x_test - reconstructions),axis=1)
reconstructions_error = (reconstructions_error - np.min(reconstructions_error)) / (np.max(reconstructions_error) - np.min(reconstructions_error))
threshold = np.percentile(reconstructions_error,46)
print(threshold)
predicated_an = np.where(reconstructions_error > threshold, 1, 0)
print(accuracy_score(y_test, predicated_an))
print(y_test,predicated_an)

plt.hist(reconstructions_error, bins=50)
plt.title('Histogram of Reconstruction Errors')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('MSLE Loss')
plt.legend(['loss', 'val_loss'])
plt.show()