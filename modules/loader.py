#loader.py: 영상, 웹캠, 이미지 불러오기 전용


import os
import cv2

# 이미지 불러오기
def load_image(image_name="study_sample_image.jpg"):
    path = os.path.join("datasets", "images", image_name)
    return cv2.imread(path)

# 비디오 파일 불러오기
def load_video(video_name="study_sample_video.avi"):
    path = os.path.join("datasets", "videos", video_name)
    return cv2.VideoCapture(path)

# 웹캠 연결 (기본값 0번 카메라)
def load_webcam(cam_index=0):
    return cv2.VideoCapture(cam_index)
