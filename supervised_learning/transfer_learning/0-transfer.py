#!/usr/bin/env python3
"""Transfer Learning"""
from tensorflow import keras as K

def preprocess_data(X, Y):
    """pre-processes the data for your model"""
    X_p = K.applications.inception_resnet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

# load CIFAR-10 dataset
(X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

# pre-process data
X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

# load InceptionResNetV2 model
base_model = K.applications.InceptionResNetV2(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling='avg'
)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Build the model
model = K.Sequential([
    K.layers.Lambda(lambda x: K.backend.resize_images(x, 7, 7, "channels_last"), input_shape=(32, 32, 3)),
    base_model,
    K.layers.Flatten(),
    K.layers.Dense(1024, activation='relu'),
    K.layers.Dropout(0.5),
    K.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_p, Y_train_p, epochs=10, validation_data=(X_test_p, Y_test_p), batch_size=32)

# Save the model
model.save('cifar10.h5')
