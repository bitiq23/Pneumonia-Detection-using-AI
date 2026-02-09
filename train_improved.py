import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

print("="*60)
print("Improved Version - Overfitting Reduction")
print("="*60)

# Paths
DATASET_PATH = r'C:\Users\AliKhalid\Desktop\pneumonia_detection\chest_xray\chest_xray'
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
VAL_PATH = os.path.join(DATASET_PATH, 'val')
TEST_PATH = os.path.join(DATASET_PATH, 'test')

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 30

# ===========================================
# Improvement 1: Stronger Data Augmentation
# ===========================================
print("\n Preparing data with stronger augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ===========================================
# Improvement 2: Lighter Model + Stronger Dropout
# ===========================================
print("\n Building improved model...")

model = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    # Dense layers (smaller)
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

# ===========================================
# Improvement 3: Smaller Learning Rate
# ===========================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)

print("\n Model summary:")
model.summary()

# ===========================================
# Improvement 4: Better Callbacks
# ===========================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=4,
    min_lr=1e-8,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model_v2.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ===========================================
# Training
# ===========================================
print("\n" + "="*60)
print(" Starting improved training...")
print("="*60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ===========================================
# Saving & Evaluation
# ===========================================
model.save('final_model_v2.h5')

print("\n Evaluating on test data...")
best_model = keras.models.load_model('best_model_v2.h5')
test_loss, test_acc, test_prec, test_rec = best_model.evaluate(test_generator)

print(f"\n Final Results:")
print(f"   Accuracy:  {test_acc*100:.2f}%")
print(f"   Precision: {test_prec*100:.2f}%")
print(f"   Recall:    {test_rec*100:.2f}%")
print(f"   F1-Score:  {2*(test_prec*test_rec)/(test_prec+test_rec):.4f}")

# ===========================================
# Plotting Results
# ===========================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(history.history['accuracy'], label='Train')
axes[0, 0].plot(history.history['val_accuracy'], label='Val')
axes[0, 0].set_title('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history.history['loss'], label='Train')
axes[0, 1].plot(history.history['val_loss'], label='Val')
axes[0, 1].set_title('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(history.history['precision'], label='Train')
axes[1, 0].plot(history.history['val_precision'], label='Val')
axes[1, 0].set_title('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(history.history['recall'], label='Train')
axes[1, 1].plot(history.history['val_recall'], label='Val')
axes[1, 1].set_title('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('improved_results.png', dpi=300)
plt.show()

print("\n Done!")
