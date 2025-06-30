# augment.py: 이미지 전처리 전용 
import cv2

# 좌우 또는 상하 반전
def flip_image(image, mode="horizontal"): 
    if mode == "horizontal":
        return cv2.flip(image, 1)
    elif mode == "vertical":
        return cv2.flip(image, 0)
    else:
        return image

# 이미지 회전
def rotate_image(image, angle=90):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0) # 회전 행렬 구하기 
    return cv2.warpAffine(image, matrix, (w, h))

# 이미지 크기 변경
def resize_image(image, width=640, height=480):
    return cv2.resize(image, (width, height))
