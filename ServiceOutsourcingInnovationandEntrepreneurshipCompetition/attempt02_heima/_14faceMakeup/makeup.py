# @Version: python3.10
# @Time: 2023/4/13 19:42
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: makeup.py
# @Software: PyCharm
# @User: chent

from PIL import Image, ImageDraw
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file('Aaron_Eckhart.jpg')

# Find all facial features
face_landmarks_list = face_recognition.face_landmarks(image)

# Create a PIL image draw object, so we can draw on the picture
pil_image = Image.fromarray(image)
for face_landmarks in face_landmarks_list:
    d = ImageDraw.Draw(pil_image, 'RGBA')

    # 画个浓眉
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(1, 1, 1, 1), width=15)
    # d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    # 嘴唇
    d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 128), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128), width=1)

    # 眼睛
    d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
    d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

    # show the pictures
    pil_image.show()