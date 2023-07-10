import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype = float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype = float)

layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Starting training...")
record = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Model trained!")

plt.xlabel("# Epoch")
plt.ylabel('Loss Magnitude')
plt.plot(record.history["loss"])

print("Let's make a prediction")
result = model.predict([100.0])
print("The result is" + str(result) + "Fahrenheit")

print("Model's Internal Variables")
print(layer.get_weights())