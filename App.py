import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image

MODEL_PATH = "best.pt"

class_names = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]
colors = np.random.uniform(0, 255, size=(len(class_names), 3))


def load_model(model_path):
    model = YOLO(model_path)
    return model


def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0] - bboxes[2] / 2, bboxes[1] - bboxes[3] / 2
    xmax, ymax = bboxes[0] + bboxes[2] / 2, bboxes[1] + bboxes[3] / 2
    return xmin, ymin, xmax, ymax


def plot_box(image, bboxes, labels):
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        xmin, ymin, xmax, ymax = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        class_name = class_names[int(labels[box_num])]

        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            color=colors[class_names.index(class_name)],
            thickness=2,
        )

        font_scale, font_thickness = min(1, max(3, int(w / 500))), min(
            2, max(10, int(w / 50))
        )
        cv2.putText(
            image,
            class_name,
            (xmin + 1, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )
    return image


def draw_traffic_light(frame, status):
    """Draw a traffic light with red, yellow, and green circles."""
    height, width = frame.shape[:2]
    x, y = 50, 100  # Top-left corner of traffic light

    colors = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}
    off_color = (50, 50, 50)  # Dark grey for inactive lights

    cv2.circle(frame, (x, y), 20, colors["red"] if status == "red" else off_color, -1)
    cv2.circle(
        frame,
        (x, y + 50),
        20,
        colors["yellow"] if status == "yellow" else off_color,
        -1,
    )
    cv2.circle(
        frame, (x, y + 100), 20, colors["green"] if status == "green" else off_color, -1
    )

    return frame


def traffic_light_simulation(video_path, model, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    out_video_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, 20.0, (frame_width, frame_height))

    frame_count = 0
    ambulance_detected = False

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            results = model(frame)
            detections = results[0].boxes

            bboxes, labels = [], []
            ambulance_detected = False

            for det in detections:
                cls = int(
                    det.cls[0].item()
                )  # Corrected access to 'cls' for detection results
                if class_names[cls] == "Ambulance":
                    ambulance_detected = True

                xywh = det.xywh[0].cpu().numpy()  # Convert tensor to numpy array
                bboxes.append(
                    [
                        xywh[0] / frame_width,
                        xywh[1] / frame_height,
                        xywh[2] / frame_width,
                        xywh[3] / frame_height,
                    ]
                )
                labels.append(cls)

            frame = plot_box(frame, bboxes, labels)

            light_status = "green" if ambulance_detected else "red"
            frame = draw_traffic_light(frame, light_status)

            status_message = (
                "Ambulance Detected!" if ambulance_detected else "No Ambulance Detected"
            )
            status_placeholder.write(status_message)

            frame_placeholder.image(frame, channels="BGR")

            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()
    return out_video_path


def detect_on_image(image, model):
    results = model(image)
    detections = results[0].boxes

    bboxes, labels = [], []
    ambulance_detected = False

    for det in detections:
        cls = int(det.cls[0].item())
        xywh = det.xywh[0].cpu().numpy()  # Convert tensor to numpy array
        h, w = image.shape[:2]
        bboxes.append([xywh[0] / w, xywh[1] / h, xywh[2] / w, xywh[3] / h])
        labels.append(cls)
        if class_names[cls] == "Ambulance":
            ambulance_detected = True

    output_image = plot_box(image, bboxes, labels)

    return output_image, ambulance_detected


def run_app():
    st.title("Ambulance Detection with Traffic Light Control")
    model = load_model(MODEL_PATH)

    uploaded_files = st.file_uploader(
        "Upload Videos or Images",
        type=["mp4", "avi", "mov", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            if uploaded_file.type.startswith("video"):
                temp_video = tempfile.NamedTemporaryFile(delete=False)
                temp_video.write(uploaded_file.read())
                temp_video.close()

                st.write(f"Processing video: {uploaded_file.name}...")
                output_video_path = traffic_light_simulation(
                    temp_video.name, model, frame_skip=5
                )

            elif uploaded_file.type.startswith("image"):
                image = Image.open(uploaded_file)
                image = np.array(image)

                st.write(f"Processing image: {uploaded_file.name}...")
                output_image, ambulance_detected = detect_on_image(image, model)

                st.image(output_image, caption=uploaded_file.name, channels="BGR")

                status_message = (
                    "Ambulance Detected!"
                    if ambulance_detected
                    else "No Ambulance Detected"
                )
                st.write(status_message)


if __name__ == "__main__":
    run_app()
