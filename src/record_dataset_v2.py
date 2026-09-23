import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# YOLLAR
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

HAND_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)

FACE_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "face_landmarker.task"
)

POSE_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "pose_landmarker_full.task"
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)


# ============================================================
# AYARLAR
# ============================================================

FPS = 30

MIN_FRAMES = 15

MAX_FRAMES = 180


# ============================================================
# YÜZ LANDMARKLARI
# ============================================================

FACE_INDICES = [

    # Sol kaş
    46, 53, 52, 65, 55,

    # Sağ kaş
    276, 283, 282, 295, 285,

    # Sol göz
    33, 133, 159, 145, 160, 144,

    # Sağ göz
    362, 263, 386, 374, 387, 373,

    # Ağız
    61, 291, 0, 17, 13, 14, 78, 308,

    # Burun / yüz merkezi
    1, 4, 5, 6
]


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
# YARDIMCI FONKSİYONLAR
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

    # Pose:
    # 11 = sol omuz
    # 12 = sağ omuz

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


def next_number(folder):

    numbers = []

    for filename in os.listdir(folder):

        if not filename.endswith(".npy"):
            continue

        name = os.path.splitext(filename)[0]

        try:
            numbers.append(int(name))

        except ValueError:
            pass

    if not numbers:
        return 1

    return max(numbers) + 1


# ============================================================
# KELİME SEÇ
# ============================================================

print()
print("========================================")
print("       TİD VERİ KAYIT V2")
print("========================================")
print()

print("1 - BEN")
print("2 - SEN")
print("3 - SEVMEK")
print("4 - MERHABA")
print("5 - TEŞEKKÜR")
print()

choice = input("Kelime seç: ").strip()


labels = {
    "1": "ben",
    "2": "sen",
    "3": "sevmek",
    "4": "merhaba",
    "5": "tesekkur"
}


if choice not in labels:

    raise ValueError(
        "Geçersiz kelime seçimi."
    )


label = labels[choice]


# ============================================================
# KLASÖR
# ============================================================

save_dir = os.path.join(
    DATASET_DIR,
    label
)

os.makedirs(
    save_dir,
    exist_ok=True
)


sample_number = next_number(
    save_dir
)


save_path = os.path.join(
    save_dir,
    f"{sample_number:03d}.npy"
)


# ============================================================
# KAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Kamera açılamadı!"
    )


print()
print(f"Kelime: {label.upper()}")
print()
print("R = kayıt başlat")
print("S = kayıt durdur ve kaydet")
print("Q = çık")
print()


# ============================================================
# DEĞİŞKENLER
# ============================================================

recording = False

frames = []

timestamp_ms = 0


# ============================================================
# ANA DÖNGÜ
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        print(
            "Kameradan görüntü alınamadı."
        )

        break


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp_ms += int(
        1000 / FPS
    )


    # ========================================================
    # EL
    # ========================================================

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


            # MediaPipe handedness
            # bilgisi kullanılıyor.

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


    # ========================================================
    # POSE
    # ========================================================

    pose = np.zeros(
        (33, 3),
        dtype=np.float32
    )


    if pose_result := pose_detector.detect_for_video(
        mp_image,
        timestamp_ms
    ):

        if pose_result.pose_landmarks:

            pose = np.array(
                [
                    landmark_xyz(lm)
                    for lm in
                    pose_result.pose_landmarks[0]
                ],
                dtype=np.float32
            )


    # ========================================================
    # YÜZ
    # ========================================================

    face = np.zeros(
        (len(FACE_INDICES), 3),
        dtype=np.float32
    )


    face_result = face_detector.detect_for_video(
        mp_image,
        timestamp_ms
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


    # ========================================================
    # BİRLEŞTİR
    # ========================================================

    combined = np.concatenate(
        [
            left_hand,
            right_hand,
            pose,
            face
        ],
        axis=0
    )


    # ========================================================
    # NORMALİZASYON
    # ========================================================

    combined = normalize_frame(
        combined
    )


    # ========================================================
    # KAYIT
    # ========================================================

    if recording:

        if len(frames) < MAX_FRAMES:

            frames.append(
                combined
            )


    # ========================================================
    # EKRAN
    # ========================================================

    if recording:

        cv2.putText(
            frame,
            "KAYIT",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            f"Frame: {len(frames)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            f"Hazir: {label.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    cv2.putText(
        frame,
        "R: Kayit | S: Kaydet | Q: Cikis",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "TID Dataset Recorder V2",
        frame
    )


    # ========================================================
    # KLAVYE
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # R
    # --------------------------------------------------------

    if key == ord("r"):

        if not recording:

            frames = []

            recording = True

            print(
                "Kayıt başladı..."
            )


    # --------------------------------------------------------
    # S
    # --------------------------------------------------------

    elif key == ord("s"):

        if recording:

            recording = False


            if len(frames) < MIN_FRAMES:

                print(
                    f"UYARI: {len(frames)} frame."
                )

                print(
                    "Kayıt çok kısa, "
                    "kaydedilmedi."
                )

                frames = []


            else:

                sequence = np.array(
                    frames,
                    dtype=np.float32
                )


                np.save(
                    save_path,
                    sequence
                )


                print()
                print(
                    "================================"
                )
                print(
                    "KAYIT BAŞARILI"
                )
                print(
                    "================================"
                )
                print(
                    f"Dosya : {save_path}"
                )
                print(
                    f"Shape : {sequence.shape}"
                )
                print(
                    f"Frame : {len(sequence)}"
                )
                print()


                break


    # --------------------------------------------------------
    # Q
    # --------------------------------------------------------

    elif key == ord("q"):

        break


# ============================================================
# TEMİZLİK
# ============================================================

cap.release()

cv2.destroyAllWindows()

hand_detector.close()
face_detector.close()
pose_detector.close()

print(
    "Program kapatıldı."
)
