import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# MODEL YOLLARI
# --------------------------------------------------

BASE_DIR = "."

HAND_MODEL = f"{BASE_DIR}/models/hand_landmarker.task"
FACE_MODEL = f"{BASE_DIR}/models/face_landmarker.task"
POSE_MODEL = f"{BASE_DIR}/models/pose_landmarker_full.task"


# --------------------------------------------------
# MEDIAPIPE AYARLARI
# --------------------------------------------------

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


hand_detector = vision.HandLandmarker.create_from_options(hand_options)
face_detector = vision.FaceLandmarker.create_from_options(face_options)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)


# --------------------------------------------------
# KAMERA
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı!")


frame_timestamp_ms = 0


print("Kamera başlatıldı.")
print("Çıkmak için Q tuşuna bas.")


while True:

    success, frame = cap.read()

    if not success:
        print("Kameradan görüntü alınamadı.")
        break

    # OpenCV: BGR
    # MediaPipe: SRGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    frame_timestamp_ms += 33


    # --------------------------------------------------
    # EL
    # --------------------------------------------------

    hand_result = hand_detector.detect_for_video(
        mp_image,
        frame_timestamp_ms,
    )


    # --------------------------------------------------
    # YÜZ
    # --------------------------------------------------

    face_result = face_detector.detect_for_video(
        mp_image,
        frame_timestamp_ms,
    )


    # --------------------------------------------------
    # VÜCUT
    # --------------------------------------------------

    pose_result = pose_detector.detect_for_video(
        mp_image,
        frame_timestamp_ms,
    )


    # --------------------------------------------------
    # EL LANDMARKLARI
    # --------------------------------------------------

    height, width, _ = frame.shape

    if hand_result.hand_landmarks:

        for hand_landmarks in hand_result.hand_landmarks:

            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1,
                )


    # --------------------------------------------------
    # YÜZ LANDMARKLARI
    # --------------------------------------------------

    if face_result.face_landmarks:

        for face_landmarks in face_result.face_landmarks:

            # Her yüz landmarkını çiziyoruz.
            # Daha sonra sadece kaş/göz/ağız bölgelerini
            # özellik olarak seçebiliriz.

            for landmark in face_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (255, 0, 0),
                    -1,
                )


    # --------------------------------------------------
    # POSE LANDMARKLARI
    # --------------------------------------------------

    if pose_result.pose_landmarks:

        for pose_landmarks in pose_result.pose_landmarks:

            for landmark in pose_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 0, 255),
                    -1,
                )


    # --------------------------------------------------
    # BİLGİ YAZISI
    # --------------------------------------------------

    hand_count = len(hand_result.hand_landmarks)
    face_count = len(face_result.face_landmarks)
    pose_count = len(pose_result.pose_landmarks)

    cv2.putText(
        frame,
        f"Hands: {hand_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Face: {face_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Pose: {pose_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )


    # --------------------------------------------------
    # EKRANA GÖSTER
    # --------------------------------------------------

    cv2.imshow(
        "Turk Isaret Dili - Landmark Test",
        frame,
    )


    # Q -> çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# TEMİZLE
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()

hand_detector.close()
face_detector.close()
pose_detector.close()

print("Program kapatildi.")
