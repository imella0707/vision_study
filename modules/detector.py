#detector.py: YOLO로 탐지 기능 전용 (image & video 통합 관리)

from ultralytics import YOLO
import cv2

# 이미지 객체 탐지
def detect_with_yolo_image(image, model_name="yolo11s.pt", confidence=0.5, title="YOLO"):
    model = YOLO(model_name)
    result = model.predict(image, conf=confidence, verbose=False)[0]
    cv2.imshow(f"{title} - {model_name} @ conf {confidence}", result.plot())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 비디오 또는 웹캠 스트림 탐지
def detect_with_yolo_video(cap, model_name="yolo11s.pt", confidence=0.5):
    model = YOLO(model_name)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = model.predict(frame, conf=confidence, verbose=False)[0]
        cv2.imshow(f"YOLO Video - {model_name}", result.plot())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
