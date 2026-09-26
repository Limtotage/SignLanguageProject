import os
import numpy as np


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

LABELS = [
    "default",
    "ben",
    "sen",
    "sevmek",
    "merhaba",
    "tesekkur",
    "evet",
    "hayir",
    "gelmek",
    "gitmek",
    "yardim",
    "ne",
    "nerede",
    "su",
    "yemek_food",
    "arkadas",
    "bugun",
    "yarin",
    "iyi",
    "kotu",
    "tekrar"
]

LABEL_TO_ID = {
    label: i
    for i, label in enumerate(LABELS)
}

TARGET_FRAMES = 90


def resample_sequence(sequence, target_frames):
    """
    Değişken uzunluktaki sekansı
    target_frames uzunluğuna getirir.
    """

    old_frames = sequence.shape[0]

    old_indices = np.linspace(
        0,
        old_frames - 1,
        num=old_frames
    )

    new_indices = np.linspace(
        0,
        old_frames - 1,
        num=target_frames
    )

    output = np.empty(
        (
            target_frames,
            sequence.shape[1],
            sequence.shape[2]
        ),
        dtype=np.float32
    )

    for landmark in range(sequence.shape[1]):

        for coord in range(sequence.shape[2]):

            output[:, landmark, coord] = np.interp(
                new_indices,
                old_indices,
                sequence[:, landmark, coord]
            )

    return output


X = []
y = []


print()
print("========================================")
print("       DATASET HAZIRLAMA")
print("========================================")
print()


for label in LABELS:

    label_dir = os.path.join(
        DATASET_DIR,
        label
    )

    label_id = LABEL_TO_ID[label]

    files = sorted(
        f for f in os.listdir(label_dir)
        if f.endswith(".npy")
    )

    print(
        f"{label.upper():10s}: "
        f"{len(files)} kayıt"
    )

    for filename in files:

        path = os.path.join(
            label_dir,
            filename
        )

        sequence = np.load(path)

        if sequence.ndim != 3:
            raise ValueError(
                f"Hatalı boyut: {path} "
                f"{sequence.shape}"
            )

        if sequence.shape[1:] != (109, 3):
            raise ValueError(
                f"Hatalı landmark boyutu: {path} "
                f"{sequence.shape}"
            )

        sequence = resample_sequence(
            sequence,
            TARGET_FRAMES
        )

        X.append(sequence)
        y.append(label_id)


X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int64
)


os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


np.save(
    os.path.join(PROCESSED_DIR, "X.npy"),
    X
)

np.save(
    os.path.join(PROCESSED_DIR, "y.npy"),
    y
)


print()
print("========================================")
print("HAZIRLAMA TAMAMLANDI")
print("========================================")
print()
print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print()
print("Sınıflar:")

for label, label_id in LABEL_TO_ID.items():

    count = np.sum(y == label_id)

    print(
        f"  {label_id} -> {label}: {count}"
    )

print()
print(
    f"Kaydedildi: {PROCESSED_DIR}"
)
