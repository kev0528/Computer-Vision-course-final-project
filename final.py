import cv2
import mediapipe as mp
import numpy as np
import os

# --- Initialize MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# thresholds
SMILE_HAPPY    = -20.0   # very strong smile
OPEN_ANGRY_MAX = 0.08    # almost closed mouth → angry
OPEN_SUR_LOW   = 0.15    # medium open → surprised
OPEN_SUR_HIGH  = 0.50    # too large → angry shout

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def analyze_emotion_geometry(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[{image_path}] Error: image not found")
        return None  # 回傳 None 代表失敗

    h, w, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_img)
    if not results.multi_face_landmarks:
        print(f"[{image_path}] No face detected.")
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    # landmark indices
    LEFT_CORNER = 61
    RIGHT_CORNER = 291
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_BROW = 65
    LEFT_EYE_TOP = 159

    def get_point(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

    p_left = get_point(LEFT_CORNER)
    p_right = get_point(RIGHT_CORNER)
    p_top = get_point(UPPER_LIP)
    p_bottom = get_point(LOWER_LIP)
    p_brow = get_point(LEFT_BROW)
    p_eye = get_point(LEFT_EYE_TOP)

    # geometric features
    corners_y = (p_left[1] + p_right[1]) / 2
    center_y = (p_top[1] + p_bottom[1]) / 2
    smile_ratio = corners_y - center_y   # more negative → more smile

    mouth_height = np.linalg.norm(p_top - p_bottom)
    mouth_width = np.linalg.norm(p_left - p_right)
    open_ratio = mouth_height / (mouth_width + 1e-6)

    brow_eye_dist = np.linalg.norm(p_brow - p_eye)
    normalized_brow_dist = brow_eye_dist / (mouth_width + 1e-6)

    # --- 5. Rule-based emotion classification (tuned) ---
    # 先預設為 Neutral
    emotion = "Neutral"

    # ===== Pain：優先判斷 =====

    # Pain 型態 A：嘴巴幾乎沒開、微負的 smile、眉毛偏高或偏低
    if (open_ratio < 0.01 and
        -2.9 < smile_ratio < -0.5 and
        (normalized_brow_dist > 0.315 or normalized_brow_dist < 0.235)):
        emotion = "Pain"

    # Pain 型態 B：嘴巴中度張開、smile > 0
    elif (0.26 < open_ratio < 0.33 and
          smile_ratio > 0.0):
        emotion = "Pain"

    # Pain 型態 C：眉毛抬很高、嘴微開
    elif (normalized_brow_dist > 0.39 and
          open_ratio < 0.04 and
          smile_ratio > -15.0):
        emotion = "Pain"

    else:
        # ===== 其餘再分 Happy / Surprised / Angry / Neutral =====

        # Happy：笑得很誇張，或是 smile 很負且嘴巴不太開
        if (smile_ratio < SMILE_HAPPY) or (smile_ratio < -7.8 and open_ratio < 0.11):
            emotion = "Happy"

        # Surprised：嘴巴張開在中間區間
        elif OPEN_SUR_LOW <= open_ratio <= OPEN_SUR_HIGH:
            emotion = "Surprised"

        # Angry：嘴巴幾乎沒開，或張超大
        elif open_ratio < OPEN_ANGRY_MAX or open_ratio > OPEN_SUR_HIGH:
            emotion = "Angry"

        # 再把一小塊「其實比較像平靜」從 Angry 改回 Neutral
        if (emotion == "Angry" and
            open_ratio < 0.003 and
            -1.6 < smile_ratio < 2.1 and
            0.25 < normalized_brow_dist < 0.34):
            emotion = "Neutral"

    return emotion, smile_ratio, open_ratio, normalized_brow_dist


def analyze_folder(folder_path):
    files = sorted(os.listdir(folder_path))
    for name in files:
        full_path = os.path.join(folder_path, name)
        if not os.path.isfile(full_path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMG_EXT:
            continue

        result = analyze_emotion_geometry(full_path)
        if result is None:
            continue
        emotion, smile, opn, brow = result
        print(f"{name}, Emotion: {emotion}, "
              f"Smile={smile:.2f}, Open={opn:.3f}, Brow={brow:.3f}")


# --- Run ---
analyze_folder("./figure2")