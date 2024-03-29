# @Version: python3.10
# @Time: 2023/4/14 14:08
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: facerec_from_video_file.py
# @Software: PyCharm
# @User: chent

import face_recognition
import cv2

# Open the input movie file
# 读入影片并得到影片长度
input_movie = cv2.VideoCapture("test.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches ir
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 第一个参数是要保存的文件的路径
# fourcc 指定编码器#
# fps 要保存的视频的帧率#
# frame size 要保存的文件的画面尺寸
# isColor 指示是黑白画面还是彩色的画面
# fourcc
# fourcc 本身是一个 32 位的无符号数值，用 4 个字母表示采用的编码器
# 常用的有“DIVX”、”MJPG”、“XVID”、“X264”。可用的列表在这里。

# 推荐使用 ”XVID”，但一般依据你的电脑环境安装了哪些编码器。
output_movie = cv2.VideoWriter('output.mp4', fourcc, 25, (64, 360))
# output_movie = cv2.VideoWriter('output.avi', fourcc， 29.97，(640，360))
# Load some sample pictures and learn how to recognize them.
lmm_image = face_recognition.load_image_file("man01.png")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

al_image = face_recognition.load_image_file("donard.png")
al_face_encoding = face_recognition.face_encodings(al_image)[0]
know_faces = [
    lmm_face_encoding,
    al_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(know_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier,
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name ="Qidian"
        elif match[1]:
            name ="Quxiaoxiao"
        face_names.append(name)

        # Label the results
    for(top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done
input_movie.release()
cv2.destroyAllWindows()