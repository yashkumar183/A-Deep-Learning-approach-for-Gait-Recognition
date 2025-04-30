# pyright: reportMissingImports=false
import tensorflow as tf
model = tf.keras.models.load_model("C:/Users/yashkumar/Desktop/Gait_Recognition/model/gait_model_cnn.keras")

IMG_SIZE = 128
NUM_CLASSES = 312

def load_and_preprocess_gei_tf(path):
    image_string = tf.io.read_file(path)
    image = tf.image.decode_png(image_string, channels=1)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)  # shape becomes (1, 128, 128, 1)
    return image
gei_path = "C:/Users/yashkumar/Desktop/Gait_Recognition/dataset_test/image_gei.png" 
img = load_and_preprocess_gei_tf(gei_path)

pred = model(img, training=False)
print("Predicted class:", tf.argmax(pred[0]).numpy())

top5 = tf.argsort(pred[0], direction='DESCENDING')[:5]
for i in top5:
    class_id = i.numpy()
    prob = pred[0][class_id].numpy()
    print(f"Class: {class_id}, Probability: {prob:.4f}")

