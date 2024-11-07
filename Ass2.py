import pandas as pd
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten,Dense,Dropout,Conv2D,MaxPooling2D
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train , x_test = x_train/255.0 , x_test /255.0

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential([Flatten(input_shape=(28,28)),
                    Dense(128,activation='relu'),
                    Dense(10,activation='softmax')])
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_test,y_test))

test_loss , test_acc = model.evaluate(x_test,y_test)
print(f"This is Test_loss:{test_loss},This is Test_acc:{test_acc}")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],label = 'Traning loss')
plt.plot(history.history['val_loss'],label='Valiadtion loos')
plt.xlabel('Epoches')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],label = 'Traing accuracy')
plt.plot(history.history['val_accuracy'],label = 'Validation accuracy')
plt.legend()

plt.tight_layout()
plt.show()


pre_val = model.predict(x_test)
plt.imshow(x_test[20])
plt.show()

print(np.argmax(pre_val[20],axis=0))