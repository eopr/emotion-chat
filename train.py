"""
Training script for emotion recognition model using existing `data/train` and `data/test` folders.
Uses standalone Keras for ICS351 compatibility. Edit the `epochs` variable to adjust training length.
"""
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# === Configuration ===
batch_size     = 64
image_size     = (48, 48)
num_classes    = 7
epochs         = 30  # <--- change this value to adjust number of training epochs

data_dir       = 'data'
train_dir      = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'test')  # Use data/test for validation
output_weights = 'model.h5'

# Verify dataset directories exist
if not os.path.isdir(train_dir) or not os.path.isdir(validation_dir):
    raise FileNotFoundError(
        f"Ensure '{train_dir}' and '{validation_dir}' exist and contain emotion subfolders."
    )

# === Data Generators ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.7, 1.3),
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

# === Model Definition ===
input_tensor = Input(shape=(48, 48, 1))
x = Concatenate()([input_tensor, input_tensor, input_tensor])
base_model = MobileNetV2(input_tensor=x, include_top=False, weights='imagenet')
# Freeze early layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Callbacks ===
callbacks = [
    ModelCheckpoint(output_weights, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# === Train ===
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# === Save Weights ===
model.save_weights(output_weights)
print(f"Training complete. Weights saved to {output_weights}")

# === Plot Training History ===
# Use actual number of epochs run (history length) to avoid mismatch if early stopping
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()