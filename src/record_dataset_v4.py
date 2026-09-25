import os
import cv2
import numpy as np
import mediapipe as mp

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

DATASET_DIR = os.path.join(
    BASE_DIR, "dataset"
)


# ========================================
# AYARLAR
# ========================================

FPS = 30

MIN_FRAMES = 15
MAX_FRAMES = 180


# ========================================
# YÜZ LANDMARKLARI
# ========================================

FACE_INDICES = [
    46, 53, 52, 65, 55,
    276, 283, 282, 295, 285,

    33, 133, 159, 145, 160, 144,

    362, 263, 386, 374, 387, 373,

    61, 291, 0, 17, 13, 14, 78, 308,

    1, 4, 5, 6
]


# ========================================
# MEDIAPIPE AYARLARI
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


hand_detector = vision.HandLandmarker.create_from_options(
    hand_options
)

face_detector = vision.FaceLandmarker.create_from_options(
    face_options
)

pose_detector = vision.PoseLandmarker.create_from_options(
    pose_options
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

    """
    combined sıralaması:

    0-20    : sol el
    21-41   : sağ el
    42-74   : pose
    75-108  : yüz
    """

    points = points.copy()

    points = np.nan_to_num(
        points,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Omuzlar
    left_shoulder = points[53]
    right_shoulder = points[54]

    # Omuz merkezi
    center = (
        left_shoulder + right_shoulder
    ) / 2.0

    points -= center

    # Omuz mesafesi
    shoulder_distance = np.linalg.norm(
        left_shoulder - right_shoulder
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


# ========================================
# LANDMARK ÇİZİM FONKSİYONLARI
# ========================================

def draw_hand_landmarks(
    frame,
    landmarks,
    connections,
    point_color,
    line_color
):

    if landmarks is None:
        return

    h, w = frame.shape[:2]

    points = []

    for lm in landmarks:

        x = int(lm[0] * w)
        y = int(lm[1] * h)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            4,
            point_color,
            -1
        )

    # Bağlantılar
    for start, end in connections:

        if start >= len(points):
            continue

        if end >= len(points):
            continue

        cv2.line(
            frame,
            points[start],
            points[end],
            line_color,
            2
        )


def draw_pose_landmarks(
    frame,
    landmarks
):

    if landmarks is None:
        return

    h, w = frame.shape[:2]

    points = []

    for lm in landmarks:

        x = int(lm[0] * w)
        y = int(lm[1] * h)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            3,
            (255, 255, 0),
            -1
        )

    # Pose bağlantıları
    pose_connections = [
        (11, 12),

        (11, 13),
        (13, 15),

        (12, 14),
        (14, 16),

        (11, 23),
        (12, 24),

        (23, 24),

        (23, 25),
        (25, 27),

        (24, 26),
        (26, 28)
    ]

    for start, end in pose_connections:

        if start >= len(points):
            continue

        if end >= len(points):
            continue

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 0),
            2
        )


def draw_face_landmarks(
    frame,
    landmarks
):

    if landmarks is None:
        return

    h, w = frame.shape[:2]

    for lm in landmarks:

        x = int(lm[0] * w)
        y = int(lm[1] * h)

        cv2.circle(
            frame,
            (x, y),
            2,
            (255, 0, 255),
            -1
        )


# ========================================
# MEDIAPIPE EL BAĞLANTILARI
# ========================================

HAND_CONNECTIONS = [

    # Başparmak
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # İşaret
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Orta
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Yüzük
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Serçe
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Avuç
    (0, 17)
]


# ========================================
# MOUSE KONTROL
# ========================================

recording = False
save_requested = False


def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global recording
    global save_requested

    # ====================================
    # SAĞ TIK → BAŞLAT
    # ====================================

    if event == cv2.EVENT_RBUTTONDOWN:

        if not recording:

            recording = True

            print()
            print("================================")
            print("KAYIT BAŞLADI")
            print("================================")
            print()


    # ====================================
    # SOL TIK → BİTİR
    # ====================================

    elif event == cv2.EVENT_LBUTTONDOWN:

        if recording:

            recording = False
            save_requested = True


# ========================================
# PROGRAM BAŞLANGICI
# ========================================

print()
print("========================================")
print("       TİD VERİ KAYIT V2")
print("========================================")
print()

print("0 - DEFAULT")
print("1 - BEN")
print("2 - SEN")
print("3 - SEVMEK")
print("4 - MERHABA")
print("5 - TEŞEKKÜR")
print()

choice = input("Kelime seç: ").strip()


labels = {
    "0": "default",
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


# ========================================
# KAYIT KLASÖRÜ
# ========================================

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


# ========================================
# KAMERA
# ========================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    raise RuntimeError(
        "Kamera açılamadı!"
    )


# ========================================
# PENCERE
# ========================================

WINDOW_NAME = "TID Dataset Recorder V2"

cv2.namedWindow(
    WINDOW_NAME
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# ========================================
# BAŞLANGIÇ
# ========================================

print()
print(
    f"Kelime: {label.upper()}"
)
print()

print("SAĞ TIK  = Kaydı başlat")
print("SOL TIK  = Kaydı bitir ve kaydet")
print("Q        = Çıkış")
print()


frames = []

timestamp_ms = 0


# ========================================
# ANA DÖNGÜ
# ========================================

while True:

    success, frame = cap.read()

    if not success:

        print(
            "Kameradan görüntü alınamadı."
        )

        break


    # ====================================
    # RGB
    # ====================================

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


    # ====================================
    # EL
    # ====================================

    hand_result = (
        hand_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )
    )


    left_hand = empty_hand()
    right_hand = empty_hand()

    left_detected = False
    right_detected = False


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
                    left_detected = True


                elif handedness == "Right":

                    right_hand = points
                    right_detected = True


    # ====================================
    # POSE
    # ====================================

    pose = np.zeros(
        (33, 3),
        dtype=np.float32
    )

    pose_detected = False


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
                for lm in pose_result
                .pose_landmarks[0]
            ],
            dtype=np.float32
        )

        pose_detected = True


    # ====================================
    # FACE
    # ====================================

    face = np.zeros(
        (len(FACE_INDICES), 3),
        dtype=np.float32
    )

    face_detected = False


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

        face_detected = True


    # ====================================
    # LANDMARKLARI EKRANA ÇİZ
    # ====================================

    # Sol el
    if left_detected:

        draw_hand_landmarks(
            frame,
            left_hand,
            HAND_CONNECTIONS,
            (0, 255, 0),
            (0, 180, 0)
        )


    # Sağ el
    if right_detected:

        draw_hand_landmarks(
            frame,
            right_hand,
            HAND_CONNECTIONS,
            (0, 0, 255),
            (0, 0, 180)
        )


    # Pose
    if pose_detected:

        draw_pose_landmarks(
            frame,
            pose
        )


    # Face
    if face_detected:

        draw_face_landmarks(
            frame,
            face
        )


    # ====================================
    # BİRLEŞTİR
    # ====================================

    combined = np.concatenate(
        [
            left_hand,
            right_hand,
            pose,
            face
        ],
        axis=0
    )


    # ====================================
    # NORMALİZASYON
    # ====================================

    combined = normalize_frame(
        combined
    )


    # ====================================
    # KAYIT
    # ====================================

    if recording:

        if len(frames) < MAX_FRAMES:

            frames.append(
                combined
            )


    # ====================================
    # ÜST BİLGİ
    # ====================================

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
            f"Frame: {len(frames)}/{MAX_FRAMES}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            f"HAZIR: {label.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ====================================
    # LANDMARK DURUMU
    # ====================================

    hand_count = int(
        left_detected
    ) + int(
        right_detected
    )


    status = (
        f"EL: {hand_count}/2"
        f" | POSE: "
        f"{'OK' if pose_detected else 'YOK'}"
        f" | YUZ: "
        f"{'OK' if face_detected else 'YOK'}"
    )


    cv2.putText(
        frame,
        status,
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ====================================
    # ALT BİLGİ
    # ====================================

    cv2.putText(
        frame,
        "SAG TIK: Baslat | SOL TIK: Kaydet | Q: Cikis",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ====================================
    # GÖSTER
    # ====================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ====================================
    # KLAVYE
    # ====================================

    key = cv2.waitKey(1) & 0xFF


    # ====================================
    # Q → ÇIKIŞ
    # ====================================

    if key == ord("q"):

        break


    # ====================================
    # KAYIT İSTEĞİ
    # ====================================

    if save_requested:

        save_requested = False


        # --------------------------------
        # ÇOK KISA KAYIT
        # --------------------------------

        if len(frames) < MIN_FRAMES:

            print(
                f"UYARI: {len(frames)} frame."
            )

            print(
                "Kayıt çok kısa, kaydedilmedi."
            )

            frames = []


        # --------------------------------
        # KAYDET
        # --------------------------------

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
            print("================================")
            print("KAYIT BAŞARILI")
            print("================================")

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


# ========================================
# TEMİZLE
# ========================================

cap.release()

cv2.destroyAllWindows()

hand_detector.close()
face_detector.close()
pose_detector.close()


print(
    "Program kapatıldı."
)
