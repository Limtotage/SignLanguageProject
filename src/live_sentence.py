import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PROJE YOLLARI
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "tid_gru.keras"
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


# ============================================================
# AYARLAR
# ============================================================

SEQUENCE_LENGTH = 90

PREDICT_EVERY = 5

CONFIDENCE_THRESHOLD = 0.70

STABLE_PREDICTIONS = 5

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


# ============================================================
# MODEL
# ============================================================

print("Model yükleniyor...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model hazır.")


# ============================================================
# MEDIAPIPE
# ============================================================

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


hand_detector = vision.HandLandmarker.create_from_options(
    hand_options
)

face_detector = vision.FaceLandmarker.create_from_options(
    face_options
)

pose_detector = vision.PoseLandmarker.create_from_options(
    pose_options
)


# ============================================================
# LANDMARK FONKSİYONLARI
# ============================================================

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


def extract_frame(mp_image, timestamp_ms):

    # --------------------------------------------------------
    # HAND
    # --------------------------------------------------------

    hand_result = hand_detector.detect_for_video(
        mp_image,
        timestamp_ms
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
                and i < len(hand_result.handedness)
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


    # --------------------------------------------------------
    # POSE
    # --------------------------------------------------------

    pose = np.zeros(
        (33, 3),
        dtype=np.float32
    )

    pose_result = pose_detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if pose_result.pose_landmarks:

        pose = np.array(
            [
                landmark_xyz(lm)
                for lm in pose_result.pose_landmarks[0]
            ],
            dtype=np.float32
        )


    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    face = np.zeros(
        (len(FACE_INDICES), 3),
        dtype=np.float32
    )

    face_result = face_detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if face_result.face_landmarks:

        face_landmarks = face_result.face_landmarks[0]

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


    # --------------------------------------------------------
    # BİRLEŞTİR
    # --------------------------------------------------------

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


# ============================================================
# TÜRKÇE CÜMLE OLUŞTURUCU
# ============================================================

def make_sentence(words):

    if not words:

        return ""


    # BEN + SEN + SEVMEK
    if words == [
        "BEN",
        "SEN",
        "SEVMEK"
    ]:

        return "Ben seni seviyorum."


    # MERHABA
    if words == ["MERHABA"]:

        return "Merhaba."


    # TESEKKUR
    if words == ["TESEKKUR"]:

        return "Teşekkür ederim."


    # BEN + TESEKKUR
    if words == [
        "BEN",
        "TESEKKUR"
    ]:

        return "Ben teşekkür ederim."


    # Genel basit dönüşüm
    result = []

    for word in words:

        if word == "BEN":

            result.append("Ben")

        elif word == "SEN":

            result.append("sen")

        elif word == "SEVMEK":

            result.append("sevmek")

        elif word == "MERHABA":

            result.append("merhaba")

        elif word == "TESEKKUR":

            result.append("teşekkür")

    if not result:

        return ""

    return " ".join(result) + "."


# ============================================================
# KAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Kamera açılamadı!"
    )


cv2.namedWindow(
    "TID Sentence"
)


# ============================================================
# BUFFERLAR
# ============================================================

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)

prediction_history = deque(
    maxlen=STABLE_PREDICTIONS
)

sentence_words = []

last_added_word = None

timestamp_ms = 0

frame_counter = 0

current_prediction = "HAZIR"

current_confidence = 0.0


# ============================================================
# ANA DÖNGÜ
# ============================================================

print()
print("========================================")
print("       TİD CÜMLE OLUŞTURUCU")
print("========================================")
print()
print("Bir işaret yap.")
print("Kararlı tahmin cümleye eklenir.")
print()
print("C = Cümleyi temizle")
print("Q = Çıkış")
print()


while True:

    success, frame = cap.read()

    if not success:

        print("Kamera görüntüsü alınamadı.")

        break


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp_ms += 33


    # --------------------------------------------------------
    # LANDMARK
    # --------------------------------------------------------

    combined = extract_frame(
        mp_image,
        timestamp_ms
    )


    sequence_buffer.append(
        combined
    )


    frame_counter += 1


    # --------------------------------------------------------
    # MODEL TAHMİNİ
    # --------------------------------------------------------

    if (
        len(sequence_buffer) == SEQUENCE_LENGTH
        and
        frame_counter % PREDICT_EVERY == 0
    ):

        sequence = np.array(
            sequence_buffer,
            dtype=np.float32
        )


        sequence = sequence.reshape(
            1,
            SEQUENCE_LENGTH,
            109 * 3
        )


        probabilities = model.predict(
            sequence,
            verbose=0
        )[0]


        class_id = int(
            np.argmax(probabilities)
        )


        current_prediction = LABELS[
            class_id
        ]


        current_confidence = float(
            probabilities[class_id]
        )


        # ----------------------------------------------------
        # GÜVEN EŞİĞİ
        # ----------------------------------------------------

        if current_confidence >= CONFIDENCE_THRESHOLD:

            prediction_history.append(
                current_prediction
            )


        # ----------------------------------------------------
        # KARARLI TAHMİN
        # ----------------------------------------------------

        if len(prediction_history) == STABLE_PREDICTIONS:

            most_common = Counter(
                prediction_history
            ).most_common(1)[0][0]


            count = Counter(
                prediction_history
            ).most_common(1)[0][1]


            # En az 4 / 5 tahmin aynıysa
            if count >= 4:

                if most_common != last_added_word:

                    sentence_words.append(
                        most_common
                    )

                    last_added_word = most_common

                    print(
                        "Kelime eklendi:",
                        most_common
                    )


                    print(
                        "Kelime dizisi:",
                        sentence_words
                    )


                    print(
                        "Cümle:",
                        make_sentence(
                            sentence_words
                        )
                    )


                prediction_history.clear()


    # ========================================================
    # EKRAN
    # ========================================================

    cv2.putText(
        frame,
        f"Tahmin: {current_prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Guven: {current_confidence * 100:.1f}%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    sentence = make_sentence(
        sentence_words
    )


    cv2.putText(
        frame,
        "CUMLE:",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        sentence[:45],
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "C: Temizle   Q: Cikis",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "TID Sentence",
        frame
    )


    # ========================================================
    # KLAVYE
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    if key == ord("c"):

        sentence_words.clear()

        prediction_history.clear()

        last_added_word = None

        print("Cümle temizlendi.")


# ============================================================
# TEMİZLE
# ============================================================

cap.release()

cv2.destroyAllWindows()

hand_detector.close()
face_detector.close()
pose_detector.close()

print()
print("Program kapatıldı.")
