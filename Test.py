import cv2
import tensorflow as tf 

Images = "Banana"

def prepare(filepath):
    img_size = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 3)

model = tf.keras.models.load_model("Fruit_CNN.model")

prediction = model.predict([prepare('Banana.jpg')])

print(prediction)
