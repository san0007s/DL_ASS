from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D,Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models,layers
from tensorflow.keras.optimizers import Adam
import numpy as np

data_folder = "caltech-101-img"

datagen = ImageDataGenerator(rescale=1./255,
                             validation_split=0.2,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

train_data = datagen.flow_from_directory(data_folder,
                                         target_size=(32,32),
                                         class_mode = 'categorical',
                                         subset='training')

val_data = datagen.flow_from_directory(data_folder,
                                       target_size=(32,32),
                                       class_mode='categorical',
                                       subset='validation')

num_classes = len(train_data.class_indices)

base_model=VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False,input_shape=(32,32,3))

for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential([base_model,
                           GlobalAveragePooling2D(),
                           Dense(512,activation='relu'),
                           Dropout(0.5),
                           Dense(num_classes,activation='softmax')])

model.compile(optimizer=Adam(learning_rate=1e-3),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_data,steps_per_epoch=20,epochs=10,validation_data=val_data)

for layer in base_model.layers[-10:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_data,steps_per_epoch=150,epochs=1,validation_data=val_data)

class_labels = list(train_data.class_indices.keys())
x_val, y_val = val_data.next()

pred = model.predict(x_val)

plt.imshow(x_val[10])
plt.show()

print(class_labels[np.argmax(pred[10],axis = 0)])
