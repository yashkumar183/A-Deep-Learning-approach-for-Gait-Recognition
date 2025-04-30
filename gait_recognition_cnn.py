# pyright: reportMissingImports=false
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# === GPU CONFIGURATION ===
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✔️ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"❌ Error enabling memory growth: {e}")

# === Paths and Constants ===
DATASET_DIR = "C:/Users/yashkumar/Desktop/Gait_Recognition/dataset_videos"
GEI_DIR = os.path.join(DATASET_DIR, "silhouettes")
IMG_SIZE = 128
NUM_CLASSES = 312

# === TensorFlow Image Preprocessing Function ===
def parse_image(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_png(image_string, channels=1)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# === Load GEI file paths and labels ===
def get_gei_paths_and_labels():
    gei_paths = []
    labels = []
    for pid in range(NUM_CLASSES):
        person_dir = os.path.join(GEI_DIR, str(pid))
        if not os.path.exists(person_dir):
            continue
        for fname in os.listdir(person_dir):
            if fname.endswith("_gei.png"):
                gei_paths.append(os.path.join(person_dir, fname))
                labels.append(pid)
    return gei_paths, tf.keras.utils.to_categorical(labels, NUM_CLASSES)

# === Load and Split Data ===
gei_paths, labels = get_gei_paths_and_labels()
X_train_paths, X_val_paths, y_train, y_val = train_test_split(gei_paths, labels, test_size=0.2, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_paths, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_paths, y_val))

train_dataset = train_dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

labels = np.argmax(y_train, axis=1)
weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(weights))

# === Model Definition ===
input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x1 = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
x1 = MaxPooling2D()(x1)
x1 = Conv2D(64, (3,3), activation='relu', padding='same')(x1)
x1 = MaxPooling2D()(x1)
x1 = Conv2D(128, (3,3), activation='relu', padding='same')(x1)
x1 = MaxPooling2D()(x1)
x1 = Conv2D(256, (3,3), activation='relu', padding='same')(x1)
x1 = MaxPooling2D()(x1)
x1 = Flatten()(x1)
x1 = Dropout(0.2)(x1)
x = Dense(256, activation='relu')(x1)
x = Dropout(0.2)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Training ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
start_time = time.time()
history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=[early_stop], class_weight=class_weights)
end_time = time.time()
print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")

# === Evaluation ===
y_true = np.argmax(np.vstack([y for _, y in val_dataset.unbatch()]), axis=1)
y_pred_probs = model.predict(val_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

correct_preds = (y_pred == y_true)
class_accuracy = {}
for class_id in np.unique(y_true):
    indices = np.where(y_true == class_id)[0]
    acc = np.mean(correct_preds[indices])
    class_accuracy[class_id] = acc

class_acc_df = pd.DataFrame(list(class_accuracy.items()), columns=["Class", "Accuracy"])
class_acc_df = class_acc_df.sort_values(by="Accuracy")
print("\nWorst performing classes:")
print(class_acc_df.head(10))

model.save("C:/Users/yashkumar/Desktop/Gait_Recognition/model/gait_model_cnn.keras")

# === Plotting ===
plt.figure(figsize=(12, 5))
plt.bar(class_acc_df["Class"].astype(str), class_acc_df["Accuracy"], color='skyblue')
plt.xticks(rotation=90)
plt.title("Per-Class Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Class ID")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

def plot_learning_curves(history):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.tight_layout()
    plt.show()

plot_learning_curves(history)
