import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ========================================
# YOLLAR
# ========================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "tid_gru.keras"
)


# ========================================
# AYARLAR
# ========================================

NUM_CLASSES = 6

RANDOM_STATE = 42


# ========================================
# VERİYİ YÜKLE
# ========================================

X = np.load(
    os.path.join(
        PROCESSED_DIR,
        "X.npy"
    )
)

y = np.load(
    os.path.join(
        PROCESSED_DIR,
        "y.npy"
    )
)


print()
print("========================================")
print("          TİD GRU EĞİTİMİ")
print("========================================")
print()

print("X:", X.shape)
print("y:", y.shape)


# ========================================
# 109 LANDMARK × 3 KOORDİNAT
# → 327 ÖZELLİK
# ========================================

X = X.reshape(
    X.shape[0],
    X.shape[1],
    X.shape[2] * X.shape[3]
)


print(
    "GRU input:",
    X.shape
)


# ========================================
# TRAIN / TEST
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


print()
print("Train:", X_train.shape)
print("Test :", X_test.shape)


# ========================================
# MODEL
# ========================================

model = Sequential([
    GRU(
        128,
        return_sequences=True,
        input_shape=(X.shape[1], X.shape[2])
    ),

    Dropout(0.3),

    GRU(64),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ========================================
# COMPILE
# ========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ========================================
# MODEL ÖZETİ
# ========================================

model.summary()


# ========================================
# EARLY STOPPING
# ========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ========================================
# EĞİTİM
# ========================================

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=100,

    batch_size=8,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ========================================
# TEST
# ========================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print()
print("========================================")
print("          TEST SONUCU")
print("========================================")
print()

print(
    f"Test loss     : {test_loss:.4f}"
)

print(
    f"Test accuracy : {test_accuracy:.4f}"
)

print()


# ========================================
# MODELİ KAYDET
# ========================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save(
    MODEL_PATH
)

print(
    f"Model kaydedildi:"
)

print(
    MODEL_PATH
)
