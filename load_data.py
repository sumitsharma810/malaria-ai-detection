import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# -------------------------------
# STEP 1: LOAD & PREPROCESS DATA
# -------------------------------

IMG_SIZE = 128   # Reduced size to save RAM

data = []
labels = []

base_path = os.path.abspath("dataset")

for category in ["Parasitized", "Uninfected"]:
    path = os.path.join(base_path, category)
    class_num = 0 if category == "Parasitized" else 1

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img_array = cv2.imread(img_path)

        if img_array is not None:
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(img_array)
            labels.append(class_num)

# Convert to numpy arrays (memory safe)
data = np.array(data, dtype=np.float32) / 255.0
labels = np.array(labels, dtype=np.int32)

print("Total images loaded:", len(data))

# -------------------------------
# STEP 2: TRAIN-TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    data, labels,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -------------------------------
# STEP 3: BUILD CNN MODEL
# -------------------------------

model = Sequential()

# Conv Block 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

# Conv Block 2
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

# Conv Block 3
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

# Flatten
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output Layer (Binary Classification)
model.add(Dense(1, activation='sigmoid'))

model.summary()

# -------------------------------
# STEP 4: COMPILE MODEL
# -------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# STEP 5: TRAIN MODEL
# -------------------------------

history = model.fit(
    X_train, y_train,
    epochs=5,              # Start with 5 epochs
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -------------------------------
# STEP 6: EVALUATE MODEL
# -------------------------------

loss, accuracy = model.evaluate(X_test, y_test)
print("Final Test Accuracy:", accuracy)

# -------------------------------
# STEP 7: SAVE MODEL
# -------------------------------

model.save("malaria_model.h5")
print("Model saved as malaria_model.h5")