import cv2
import mediapipe as mp
import math
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# ==========================================
# Page
# ==========================================

st.set_page_config(
    page_title="Virtual Try-On Spectacles",
    page_icon="👓",
    layout="centered"
)

st.title("👓 Virtual Try-On Spectacles")
st.write("Allow camera permission and try different glasses!")


# ==========================================
# Load Glasses
# ==========================================

glasses_list = {
    "White Glasses": cv2.imread(
        "models/glasses.png",
        cv2.IMREAD_UNCHANGED
    ),

    "Black Glasses": cv2.imread(
        "models/glasses_black.png",
        cv2.IMREAD_UNCHANGED
    ),

    "Round Glasses": cv2.imread(
        "models/glasses_round.png",
        cv2.IMREAD_UNCHANGED
    )
}


# Check images
for name, image in glasses_list.items():
    if image is None:
        st.warning(f"{name} image nahi mili!")


# ==========================================
# Glass Selection
# ==========================================

selected_glasses = st.selectbox(
    "Choose Glasses",
    list(glasses_list.keys())
)

current_glasses = glasses_list[selected_glasses]


# ==========================================
# Rotate Image
# ==========================================

def rotate_image(image, angle):

    h, w = image.shape[:2]

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return rotated


# ==========================================
# Overlay Transparent Glasses
# ==========================================

def overlay_glasses(frame, overlay, x, y):

    if overlay is None:
        return

    h, w = overlay.shape[:2]

    frame_h, frame_w = frame.shape[:2]

    # Left boundary
    if x < 0:

        overlay = overlay[:, -x:]

        w = overlay.shape[1]

        x = 0

    # Top boundary
    if y < 0:

        overlay = overlay[-y:, :]

        h = overlay.shape[0]

        y = 0

    # Right boundary
    if x + w > frame_w:

        overlay = overlay[:, :frame_w - x]

        w = overlay.shape[1]

    # Bottom boundary
    if y + h > frame_h:

        overlay = overlay[:frame_h - y, :]

        h = overlay.shape[0]

    if w <= 0 or h <= 0:
        return

    # Transparent PNG
    if overlay.shape[2] == 4:

        alpha = overlay[:, :, 3] / 255.0

        for c in range(3):

            frame[
                y:y + h,
                x:x + w,
                c
            ] = (
                alpha * overlay[:, :, c]
                +
                (1 - alpha)
                * frame[
                    y:y + h,
                    x:x + w,
                    c
                ]
            )

    else:

        frame[
            y:y + h,
            x:x + w
        ] = overlay[:, :, :3]


# ==========================================
# Video Processor
# ==========================================

class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Mirror camera
        img = cv2.flip(img, 1)

        frame_height, frame_width = img.shape[:2]

        # Convert to RGB
        rgb_frame = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # Face detection
        results = self.face_mesh.process(rgb_frame)

        # Face found
        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                # Eye landmarks
                left_eye = face_landmarks.landmark[33]

                right_eye = face_landmarks.landmark[263]

                left_x = int(
                    left_eye.x * frame_width
                )

                left_y = int(
                    left_eye.y * frame_height
                )

                right_x = int(
                    right_eye.x * frame_width
                )

                right_y = int(
                    right_eye.y * frame_height
                )

                # Distance between eyes
                eye_distance = math.sqrt(
                    (right_x - left_x) ** 2
                    +
                    (right_y - left_y) ** 2
                )

                # Glasses width
                glasses_width = int(
                    eye_distance * 2.20
                )

                if glasses_width <= 0:
                    continue

                # Original glasses size
                original_height, original_width = (
                    current_glasses.shape[:2]
                )

                # Glasses height
                glasses_height = int(
                    glasses_width
                    * original_height
                    / original_width
                )

                if glasses_height <= 0:
                    continue

                # Resize
                resized_glasses = cv2.resize(
                    current_glasses,
                    (
                        glasses_width,
                        glasses_height
                    ),
                    interpolation=cv2.INTER_AREA
                )

                # Head tilt
                angle = math.degrees(
                    math.atan2(
                        right_y - left_y,
                        right_x - left_x
                    )
                )

                # Rotate
                rotated_glasses = rotate_image(
                    resized_glasses,
                    -angle
                )

                # Center between eyes
                center_x = int(
                    (left_x + right_x) / 2
                )

                center_y = int(
                    (left_y + right_y) / 2
                )

                # Move glasses slightly down
                center_y += int(
                    glasses_height * 0.03
                )

                # Position
                rotated_height, rotated_width = (
                    rotated_glasses.shape[:2]
                )

                x = int(
                    center_x - rotated_width / 2
                )

                y = int(
                    center_y - rotated_height / 2
                )

                # Overlay
                overlay_glasses(
                    img,
                    rotated_glasses,
                    x,
                    y
                )

                cv2.putText(
                    img,
                    "Virtual Try-On",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        else:

            cv2.putText(
                img,
                "Face Not Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ==========================================
# Start Camera
# ==========================================

webrtc_streamer(
    key="virtual-try-on",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)