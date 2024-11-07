from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

x_train ,x_test= x_train/255.0,x_test/255.0

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential([Flatten(input_shape=(32,32,3)),
                    Dense(128,activation='relu'),
                    Dense(64,activation='relu'),
                    Dense(10,activation='softmax')])
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data = (x_test,y_test))

test_loss,test_acc = model.evaluate(x_test,y_test)
print(f"This is test loss:{test_loss},This is test_acc:{test_acc}")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],label = "training_loss")
plt.plot(history.history['val_loss'],label = 'validation_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],label = 'training_accuracy')
plt.plot(history.history['val_accuracy'],label= 'validation_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predict_val = model.predict(x_test)
plt.imshow(x_test[21])
plt.show()

print(class_name[np.argmax(predict_val[21],axis=0)])