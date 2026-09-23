import os
import time
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# AYARLAR
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

HAND_MODEL = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
FACE_MODEL = os.path.join(BASE_DIR, "models", "face_landmarker.task")
POSE_MODEL = os.path.join(BASE_DIR, "models", "pose_landmarker_full.task")

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

FPS = 30


# ============================================================
# YÜZ BÖLGELERİ
# ============================================================
#
# Başlangıçta yüzün tamamını kaydetmek yerine:
# - kaşlar
# - gözler
# - ağız
#
# bölgelerini kullanıyoruz.
#
# MediaPipe Face Landmarker'da bu bölgelerden yeterli
# landmark seçiyoruz.
#
# Bu liste daha sonra gerektiğinde genişletilebilir.
# ============================================================

FACE_INDICES = [
    # Sol kaş
    46, 53, 52, 65, 55,

    # Sağ kaş
    276, 283, 282, 295, 285,

    # Sol göz çevresi
    33, 133, 159, 145, 160, 144,

    # Sağ göz çevresi
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
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)


face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)


pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=vision.RunningMode.VIDEO,
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

def landmark_to_xyz(landmark):
    return [
        landmark.x,
        landmark.y,
        landmark.z
    ]


def empty_hand():
    return np.zeros((21, 3), dtype=np.float32)


def normalize_frame(points):
    """
    Frame içindeki koordinatları normalize eder.

    Merkez:
        Pose'un omuz merkezine göre

    Ölçek:
        Omuzlar arasındaki mesafeye göre

    Böylece kameraya yaklaşma/uzaklaşmanın etkisi azalır.
    """

    points = points.copy()

    # NaN / inf temizliği
    points = np.nan_to_num(
        points,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Pose landmarkları:
    # MediaPipe Pose:
    # 11 = sol omuz
    # 12 = sağ omuz

    left_shoulder = points[42]
    right_shoulder = points[63]

    center = (left_shoulder + right_shoulder) / 2.0

    points -= center

    shoulder_distance = np.linalg.norm(
        left_shoulder - right_shoulder
    )

    if shoulder_distance > 1e-6:
        points /= shoulder_distance

    return points


def get_next_sample_number(folder):
    """
    Klasördeki mevcut .npy dosyalarını kontrol eder
    ve sıradaki numarayı verir.
    """

    existing = []

    for filename in os.listdir(folder):

        if filename.endswith(".npy"):

            try:
                number = int(
                    os.path.splitext(filename)[0]
                )

                existing.append(number)

            except ValueError:
                pass

    if not existing:
        return 1

    return max(existing) + 1


# ============================================================
# KELİME SEÇ
# ============================================================

print()
print("==========================================")
print(" TİD VERİ KAYIT SİSTEMİ")
print("==========================================")
print()
print("Kelime seç:")
print()
print("1 - BEN")
print("2 - SEN")
print("3 - SEVMEK")
print("4 - MERHABA")
print("5 - TEŞEKKÜR")
print()

choice = input("Seçim: ").strip()

labels = {
    "1": "ben",
    "2": "sen",
    "3": "sevmek",
    "4": "merhaba",
    "5": "tesekkur",
}

if choice not in labels:
    raise ValueError("Geçersiz seçim.")

label = labels[choice]

save_dir = os.path.join(
    DATASET_DIR,
    label
)

os.makedirs(
    save_dir,
    exist_ok=True
)

sample_number = get_next_sample_number(
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
    raise RuntimeError("Kamera açılamadı!")


print()
print(f"Kelime: {label.upper()}")
print()
print("R = kayıt başlat")
print("S = kayıt durdur ve kaydet")
print("Q = çık")
print()


# ============================================================
# DURUM
# ============================================================

recording = False
frames = []

timestamp_ms = 0

last_time = time.time()


# ============================================================
# ANA DÖNGÜ
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Kameradan görüntü alınamadı.")
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms += int(1000 / FPS)


    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    hand_result = hand_detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    face_result = face_detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    pose_result = pose_detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # --------------------------------------------------------
    # FRAME VERİSİ
    # --------------------------------------------------------

    left_hand = empty_hand()
    right_hand = empty_hand()


    # --------------------------------------------------------
    # ELLER
    # --------------------------------------------------------

    if hand_result.hand_landmarks:

        for hand_index, hand_landmarks in enumerate(
            hand_result.hand_landmarks
        ):

            points = np.array(
                [
                    landmark_to_xyz(lm)
                    for lm in hand_landmarks
                ],
                dtype=np.float32
            )

            if hand_index == 0:
                left_hand = points

            elif hand_index == 1:
                right_hand = points


    # --------------------------------------------------------
    # POSE
    # --------------------------------------------------------

    pose = np.zeros(
        (33, 3),
        dtype=np.float32
    )

    if pose_result.pose_landmarks:

        pose_landmarks = pose_result.pose_landmarks[0]

        pose = np.array(
            [
                landmark_to_xyz(lm)
                for lm in pose_landmarks
            ],
            dtype=np.float32
        )


    # --------------------------------------------------------
    # YÜZ
    # --------------------------------------------------------

    face = np.zeros(
        (len(FACE_INDICES), 3),
        dtype=np.float32
    )

    if face_result.face_landmarks:

        face_landmarks = face_result.face_landmarks[0]

        selected = []

        for index in FACE_INDICES:

            if index < len(face_landmarks):

                selected.append(
                    landmark_to_xyz(
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
    # HEPSİNİ BİRLEŞTİR
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


    # --------------------------------------------------------
    # NORMALİZASYON
    # --------------------------------------------------------

    combined = normalize_frame(
        combined
    )


    # --------------------------------------------------------
    # KAYIT
    # --------------------------------------------------------

    if recording:

        frames.append(
            combined
        )


    # --------------------------------------------------------
    # GÖRSEL
    # --------------------------------------------------------

    if recording:

        cv2.putText(
            frame,
            "KAYIT YAPILIYOR",
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
            f"Hazir - {label.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    cv2.putText(
        frame,
        "R: Kayit  S: Durdur  Q: Cikis",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "TID Veri Kayit",
        frame
    )


    # --------------------------------------------------------
    # KLAVYE
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # R
    if key == ord("r"):

        if not recording:

            frames = []
            recording = True

            print("Kayıt başladı...")


    # S
    elif key == ord("s"):

        if recording:

            recording = False

            if len(frames) < 10:

                print(
                    "UYARI: Çok az frame kaydedildi."
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
                print("KAYIT TAMAMLANDI")
                print(
                    f"Dosya: {save_path}"
                )
                print(
                    f"Shape: {sequence.shape}"
                )
                print()

                break


    # Q
    elif key == ord("q"):

        break


# ============================================================
# TEMİZLE
# ============================================================

cap.release()
cv2.destroyAllWindows()

hand_detector.close()
face_detector.close()
pose_detector.close()

print("Program kapatıldı.")
