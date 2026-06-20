from tensorflow.keras.models import load_model
import tensorflow as tf

model = load_model("RubberLeafDiseasesModel1.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print("model converted")

with open('RubberLeafDiseasesModel1.tflite', 'wb') as f:
	f.write(tflite_model)