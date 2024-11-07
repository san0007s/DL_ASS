from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

x_train,x_test = x_train/255.0,x_test/255.0

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential([Conv2D(32,kernel_size= (3,3),activation='relu',input_shape=(32,32,3)),
                    MaxPooling2D(pool_size=(2,2)),
                    Conv2D(64,kernel_size =(3,3),activation='relu'),
                    MaxPooling2D(pool_size=(2,2)),
                    Flatten(),
                    Dense(128,activation='relu'),
                    Dense(10,activation='softmax')
                    ])

model.compile(optimizer='sgd',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data = (x_test,y_test))

test_loss,test_acc = model.evaluate(x_test,y_test)
print(f"This is test_loss:{test_loss},This is test_acc:{test_acc}")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],label='Training_loss')
plt.plot(history.history['val_loss'],label='validation_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],label='Training_Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation_Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predict_value = model.predict(x_test)
plt.imshow(x_test[21])
plt.show()

print(class_name[np.argmax(predict_value[21],axis=0)])