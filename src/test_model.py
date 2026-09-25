import os
import numpy as np
import tensorflow as tf

from collections import Counter


# ============================================================
# YOLLAR
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

X_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "X.npy"
)

Y_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "y.npy"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tid_gru.keras"
)


# ============================================================
# SINIFLAR
# ============================================================

LABELS = [
    "DEFAULT",
    "BEN",
    "SEN",
    "SEVMEK",
    "MERHABA",
    "TESEKKUR"
]


# ============================================================
# VERİ
# ============================================================

X = np.load(X_PATH)
y = np.load(Y_PATH)

print()
print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# MODEL
# ============================================================

print()
print("Model yükleniyor...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model hazır.")


# ============================================================
# ŞEKİL
# ============================================================

X_input = X.reshape(
    X.shape[0],
    X.shape[1],
    X.shape[2] * X.shape[3]
)


# ============================================================
# TAHMİN
# ============================================================

probabilities = model.predict(
    X_input,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# SINIF BAZLI SONUÇ
# ============================================================

print()
print("========================================")
print("         SINIF BAZLI TEST")
print("========================================")
print()


for class_id, label in enumerate(LABELS):

    indices = np.where(
        y == class_id
    )[0]

    true_labels = []
    predicted_labels = []

    for index in indices:

        true_labels.append(
            label
        )

        predicted_labels.append(
            LABELS[
                predictions[index]
            ]
        )


    counter = Counter(
        predicted_labels
    )


    print(
        f"{label:10s} -> "
        f"{len(indices)} örnek"
    )

    for prediction, count in counter.most_common():

        print(
            f"    {prediction:10s}: "
            f"{count}"
        )

    print()


# ============================================================
# ÖRNEKLER
# ============================================================

print()
print("========================================")
print("        TÜM ÖRNEKLER")
print("========================================")
print()


for i in range(len(X)):

    true_label = LABELS[
        y[i]
    ]

    predicted_label = LABELS[
        predictions[i]
    ]

    confidence = probabilities[i][
        predictions[i]
    ]

    status = "OK"

    if true_label != predicted_label:

        status = "YANLIŞ"


    print(
        f"{i:02d} | "
        f"Gerçek: {true_label:10s} | "
        f"Tahmin: {predicted_label:10s} | "
        f"Güven: {confidence * 100:6.2f}% | "
        f"{status}"
    )
