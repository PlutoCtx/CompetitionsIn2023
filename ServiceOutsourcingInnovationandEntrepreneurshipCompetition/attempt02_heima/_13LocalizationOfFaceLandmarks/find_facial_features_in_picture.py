# @Version: python3.10
# @Time: 2023/4/13 19:42
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: find_facial_features_in_picture.py
# @Software: PyCharm
# @User: chent

from PIL import Image, ImageDraw
import face_recognition

image = face_recognition.load_image_file('Aaron_Eckhart.jpg')
face_landmarks_list = face_recognition.face_landmarks(image)

print('There are {} face(s) in this picture'.format(len(face_landmarks_list)))

# Create a PIL image draw object, so we can draw on the picture
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:

    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        print('The {} in this face has the following points: {}'.format(facial_feature, face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=5)

# show the pictures
pil_image.show()