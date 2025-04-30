# pyright: reportMissingImports=false
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
import pandas as pd
import time
import ctypes
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Bidirectional, Concatenate, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.python.client import device_lib

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✔️ Memory growth enabled for GPU")
    except RuntimeError as e:
        print(f"❌ Error enabling memory growth: {e}")

print("\n=== List of Available Devices ===")
for device in device_lib.list_local_devices():
    print(f"{device.name} | {device.device_type} | {device.physical_device_desc}")

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Details:", tf.config.list_physical_devices('GPU'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

# === Paths and Constants ===
DATASET_DIR = "C:/Users/yashkumar/Desktop/Gait_Recognition/dataset_videos"
GEI_DIR = os.path.join(DATASET_DIR, "silhouettes")
SKELETON_DIR = os.path.join(DATASET_DIR, "skeletons")
SMPL_DIR = os.path.join(DATASET_DIR, "smpl")

IMG_SIZE = 128
NUM_FRAMES = 58
NUM_CLASSES = 312

# === Helper Functions ===
def load_gei(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load GEI image at {path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype('float32') / 255.0

def load_skeleton(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Skeleton file not found: {path}")
    with open(path, 'r') as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"Skeleton file is empty: {path}")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decoding failed for {path}: {e}")

    keypoints = []
    for i, frame in enumerate(data):
        if i >= NUM_FRAMES:
            break
        coords = []
        if not isinstance(frame, dict) or "keypoints" not in frame:
            raise ValueError(f"Invalid structure in frame {i} of {path}")
        kps = frame["keypoints"]
        
        if isinstance(kps, list):
            if all(isinstance(val, (int, float)) for val in kps):
                coords.extend(kps)
            elif all(isinstance(kp, dict) and "x" in kp and "y" in kp for kp in kps):
                coords.extend([v for kp in kps for v in (kp["x"], kp["y"])])
            elif all(isinstance(kp, (list, tuple)) and len(kp) >= 2 for kp in kps):
                coords.extend([v for kp in kps for v in kp[:2]])
            else:
                raise ValueError(f"Unrecognized keypoint format at frame {i} in {path}")
        else:
            raise ValueError(f"Expected list for keypoints but got {type(kps)} in frame {i} of {path}")

        keypoints.append(coords)

    skeleton_array = np.array(keypoints)
    if skeleton_array.shape[0] > NUM_FRAMES:
        skeleton_array = skeleton_array[:NUM_FRAMES]
    elif skeleton_array.shape[0] < NUM_FRAMES:
        pad_width = NUM_FRAMES - skeleton_array.shape[0]
        skeleton_array = np.pad(skeleton_array, ((0, pad_width), (0, 0)), mode='constant')
    return skeleton_array


def load_smpl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"SMPL file not found: {path}")
    data = np.load(path)
    pose = data.get('pose', np.zeros((0, 72)))
    joints = data.get('joints', np.zeros((0, 72)))
    
    # Truncate or pad both
    min_len = min(pose.shape[0], joints.shape[0])
    pose, joints = pose[:min_len], joints[:min_len]
    smpl = np.concatenate([pose, joints], axis=-1)

    if smpl.shape[0] > NUM_FRAMES:
        smpl = smpl[:NUM_FRAMES]
    elif smpl.shape[0] < NUM_FRAMES:
        pad_len = NUM_FRAMES - smpl.shape[0]
        smpl = np.pad(smpl, ((0, pad_len), (0, 0)), mode='constant')
    return smpl


# === Data Loader ===
def load_data():
    X_gei, X_skel_smpl, y = [], [], []
    for pid in range(NUM_CLASSES):
        person_id = f"{pid}"
        person_gei_dir = os.path.join(GEI_DIR, person_id)
        if not os.path.exists(person_gei_dir):
            continue
        for fname in os.listdir(person_gei_dir):
            if not fname.endswith("_gei.png"):
                continue
            seq = fname.replace("_gei.png", "")
            gei_path = os.path.join(person_gei_dir, fname)
            skel_path = os.path.join(SKELETON_DIR, person_id, f"{seq}.json")
            smpl_path = os.path.join(SMPL_DIR, person_id, f"{seq}.npz")
            if not (os.path.exists(gei_path) and os.path.exists(skel_path) and os.path.exists(smpl_path)):
                continue
            try:
                gei = load_gei(gei_path).reshape((IMG_SIZE, IMG_SIZE, 1))
                skeleton = load_skeleton(skel_path)
                smpl = load_smpl(smpl_path)
                fused = np.concatenate([skeleton, smpl], axis=-1)
                X_gei.append(gei)
                X_skel_smpl.append(fused)
                y.append(pid)
            except Exception as e:
                print(f"[Warning] Skipped {seq} for person {pid}: {e}")
    return np.array(X_gei), np.array(X_skel_smpl), to_categorical(y, NUM_CLASSES)

# === Load and Split Data ===
X_gei, X_seq, y = load_data()
print(f"GEI shape: {X_gei.shape}, Sequence shape: {X_seq.shape}, Labels shape: {y.shape}")

X_gei_train, X_gei_val, X_seq_train, X_seq_val, y_train, y_val = train_test_split(
    X_gei, X_seq, y, test_size=0.2, random_state=42
)


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === Model Definition ===
# GEI branch
gei_input = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x1 = Conv2D(32, (3,3), activation='relu', padding='same')(gei_input)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D()(x1)
x1 = Conv2D(64, (3,3), activation='relu', padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D()(x1)
x1 = Flatten()(x1)

# Sequence branch
seq_input = Input(shape=(NUM_FRAMES, X_seq.shape[-1]))
x2 = Bidirectional(LSTM(128, return_sequences=True))(seq_input)
x2 = Bidirectional(LSTM(128))(x2)

# Fusion
merged = Concatenate()([x1, x2])
x = Dense(256, activation='relu')(merged)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=[gei_input, seq_input], outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Now you can call model.summary()
model.summary()

# Start training
start_time = time.time()
print("start training")


# === Training ===
history = model.fit(
    [X_gei_train, X_seq_train], y_train,
    validation_data=([X_gei_val, X_seq_val], y_val),
    epochs=30, batch_size=32, callbacks=[early_stop]
)

print("training completed")
# End training
end_time = time.time()

# Calculate time taken to train
training_time = end_time - start_time
print(f"Total training time: {training_time/60:.2f} minutes")


# Get predictions
y_pred = model.predict([X_gei_val, X_seq_val])
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

correct_preds = (y_pred_labels == y_true)
class_accuracy = {}

for class_id in np.unique(y_true):
    indices = np.where(y_true == class_id)[0]
    acc = np.mean(correct_preds[indices])
    class_accuracy[class_id] = acc

# Convert to DataFrame and sort
class_acc_df = pd.DataFrame(list(class_accuracy.items()), columns=["Class", "Accuracy"])
class_acc_df = class_acc_df.sort_values(by="Accuracy", ascending=True)

# Print the 10 worst performing classes
print("\nWorst performing classes:")
print(class_acc_df.head(10))

model.save("C:/Users/yashkumar/Desktop/Gait_Recognition/model/gait_model.keras")

# === Plotting ===
# Confusion matrix
cm = confusion_matrix(y_true, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', values_format='d')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Plot per-class accuracy
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
