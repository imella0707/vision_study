# main.py: 전체 실행 흐름 제어 (조립 공장)

from modules.loader import load_image, load_video, load_webcam
from modules.detector import detect_with_yolo_video, detect_with_yolo_image
from modules.augment import flip_image, rotate_image, resize_image
import cv2


def run_image_mode():
    # 이미지 로드 및 조작 실습
    image = load_image("study_sample_image.jpg") # loader.py
    flipped = flip_image(image, mode="horizontal") #augment.py
    rotated = rotate_image(image, angle=90)
    resized = resize_image(image, width=320, height=240)

    # 모델 및 신뢰도 설정
    model_name = "yolo11s.pt"
    confidence = 0.5

    # 이미지에 대해 YOLO 탐지 실행 detector.py
    detect_with_yolo_image(flipped, model_name, confidence, title="Flipped")
    detect_with_yolo_image(rotated, model_name, confidence, title="Rotated")
    detect_with_yolo_image(resized, model_name, confidence, title="Resized")


def run_video_mode():
    # 비디오 파일 로드
    cap = load_video("study_sample_video.avi")
    model_name = "yolo11s-pose.pt"
    confidence = 0.5
    # YOLO 탐지 실행
    detect_with_yolo_video(cap, model_name, confidence)


def run_webcam_mode():
    # 웹캠 연결
    cap = load_webcam(cam_index=0)
    model_name = "yolo11s.pt"
    confidence = 0.5
    # YOLO 탐지 실행
    detect_with_yolo_video(cap, model_name, confidence)


def main():
    # 실행 모드 선택: "image", "video", "webcam"
    mode = "video"

    if mode == "image":
        run_image_mode()
    elif mode == "video":
        run_video_mode()
    elif mode == "webcam":
        run_webcam_mode()
    else:
        print("❗ 잘못된 모드입니다. 'image', 'video', 'webcam' 중 하나를 선택하세요.")


if __name__ == "__main__":
    main()
