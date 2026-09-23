import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ========================================
# PROJE YOLLARI
# ========================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

HAND_MODEL = os.path.join(
    BASE_DIR, "models", "hand_landmarker.task"
)

FACE_MODEL = os.path.join(
    BASE_DIR, "models", "face_landmarker.task"
)

POSE_MODEL = os.path.join(
    BASE_DIR, "models", "pose_landmarker_full.task"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "tid_gru.keras"
)


# ========================================
# AYARLAR
# ========================================

SEQUENCE_LENGTH = 90

PREDICT_EVERY = 5

LABELS = [
    "BEN",
    "SEN",
    "SEVMEK",
    "MERHABA",
    "TESEKKUR"
]


FACE_INDICES = [
    46, 53, 52, 65, 55,
    276, 283, 282, 295, 285,
    33, 133, 159, 145, 160, 144,
    362, 263, 386, 374, 387, 373,
    61, 291, 0, 17, 13, 14, 78, 308,
    1, 4, 5, 6
]


# ========================================
# MODEL
# ========================================

print()
print("Model yükleniyor...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model hazır.")
print()


# ========================================
# MEDIAPIPE
# ========================================

BaseOptions = python.BaseOptions


hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)


face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_MODEL
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)


pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL
    ),
    running_mode=vision.RunningMode.VIDEO
)


hand_detector = (
    vision.HandLandmarker.create_from_options(
        hand_options
    )
)

face_detector = (
    vision.FaceLandmarker.create_from_options(
        face_options
    )
)

pose_detector = (
    vision.PoseLandmarker.create_from_options(
        pose_options
    )
)


# ========================================
# YARDIMCI FONKSİYONLAR
# ========================================

def landmark_xyz(landmark):

    return [
        landmark.x,
        landmark.y,
        landmark.z
    ]


def empty_hand():

    return np.zeros(
        (21, 3),
        dtype=np.float32
    )


def normalize_frame(points):

    points = points.copy()

    points = np.nan_to_num(
        points,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # combined içindeki doğru omuz indeksleri
    left_shoulder = points[53]
    right_shoulder = points[54]

    center = (
        left_shoulder +
        right_shoulder
    ) / 2.0

    points -= center

    shoulder_distance = np.linalg.norm(
        left_shoulder -
        right_shoulder
    )

    if shoulder_distance > 1e-6:

        points /= shoulder_distance

    return points


# ========================================
# TEK FRAME LANDMARK
# ========================================

def extract_frame(
    mp_image,
    timestamp_ms
):

    # ------------------------------------
    # EL
    # ------------------------------------

    hand_result = (
        hand_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )
    )

    left_hand = empty_hand()
    right_hand = empty_hand()


    if hand_result.hand_landmarks:

        for i, hand_landmarks in enumerate(
            hand_result.hand_landmarks
        ):

            points = np.array(
                [
                    landmark_xyz(lm)
                    for lm in hand_landmarks
                ],
                dtype=np.float32
            )


            if (
                hand_result.handedness
                and i < len(
                    hand_result.handedness
                )
            ):

                handedness = (
                    hand_result
                    .handedness[i][0]
                    .category_name
                )


                if handedness == "Left":

                    left_hand = points

                elif handedness == "Right":

                    right_hand = points


    # ------------------------------------
    # POSE
    # ------------------------------------

    pose = np.zeros(
        (33, 3),
        dtype=np.float32
    )


    pose_result = (
        pose_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )
    )


    if pose_result.pose_landmarks:

        pose = np.array(
            [
                landmark_xyz(lm)
                for lm in
                pose_result.pose_landmarks[0]
            ],
            dtype=np.float32
        )


    # ------------------------------------
    # FACE
    # ------------------------------------

    face = np.zeros(
        (len(FACE_INDICES), 3),
        dtype=np.float32
    )


    face_result = (
        face_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )
    )


    if face_result.face_landmarks:

        face_landmarks = (
            face_result.face_landmarks[0]
        )

        selected = []

        for index in FACE_INDICES:

            if index < len(face_landmarks):

                selected.append(
                    landmark_xyz(
                        face_landmarks[index]
                    )
                )

            else:

                selected.append(
                    [0.0, 0.0, 0.0]
                )


        face = np.array(
            selected,
            dtype=np.float32
        )


    # ------------------------------------
    # BİRLEŞTİR
    # ------------------------------------

    combined = np.concatenate(
        [
            left_hand,
            right_hand,
            pose,
            face
        ],
        axis=0
    )


    combined = normalize_frame(
        combined
    )


    return combined


# ========================================
# KAMERA
# ========================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    raise RuntimeError(
        "Kamera açılamadı!"
    )


WINDOW_NAME = "TID Live Prediction"


cv2.namedWindow(
    WINDOW_NAME
)


# ========================================
# BUFFER
# ========================================

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)


timestamp_ms = 0

frame_counter = 0


prediction_label = "HAZIR"

prediction_confidence = 0.0


# ========================================
# ANA DÖNGÜ
# ========================================

print()
print("========================================")
print("       TİD CANLI TAHMİN")
print("========================================")
print()
print("Kamera hazır.")
print("Bir işareti yaklaşık 3 saniye yap.")
print()
print("Q = Çıkış")
print()


while True:

    success, frame = cap.read()


    if not success:

        print(
            "Kameradan görüntü alınamadı."
        )

        break


    # ------------------------------------
    # RGB
    # ------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp_ms += 33


    # ------------------------------------
    # LANDMARK
    # ------------------------------------

    combined = extract_frame(
        mp_image,
        timestamp_ms
    )


    sequence_buffer.append(
        combined
    )


    frame_counter += 1


    # ------------------------------------
    # TAHMİN
    # ------------------------------------

    if (
        len(sequence_buffer)
        == SEQUENCE_LENGTH
        and
        frame_counter % PREDICT_EVERY == 0
    ):

        sequence = np.array(
            sequence_buffer,
            dtype=np.float32
        )


        # (90, 109, 3)
        # →
        # (90, 327)

        sequence = sequence.reshape(
            1,
            SEQUENCE_LENGTH,
            109 * 3
        )


        probabilities = (
            model.predict(
                sequence,
                verbose=0
            )[0]
        )


        class_id = int(
            np.argmax(probabilities)
        )


        prediction_label = LABELS[
            class_id
        ]


        prediction_confidence = (
            float(
                probabilities[class_id]
            ) * 100.0
        )


    # ------------------------------------
    # EKRAN
    # ------------------------------------

    cv2.putText(
        frame,
        f"Tahmin: {prediction_label}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3
    )


    cv2.putText(
        frame,
        f"Guven: {prediction_confidence:.1f}%",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Q: Cikis",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ------------------------------------
    # ÇIKIŞ
    # ------------------------------------

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ========================================
# TEMİZLE
# ========================================

cap.release()

cv2.destroyAllWindows()

hand_detector.close()
face_detector.close()
pose_detector.close()


print()
print("Program kapatıldı.")
