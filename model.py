import tensorflow as tf
import numpy as np

input_dim=2
num_classes=2
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# Define the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=1)

# Evaluate the model
model.evaluate(x_train, y_train)
