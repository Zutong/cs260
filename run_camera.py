
import numpy as np
import sys
import tensorflow as tf
import cv2
import os
from sklearn.model_selection import train_test_split
import random
import sys
import urllib
from train_model import panda_joke_cnn
from train_model import is_my_face
import cv2
import urllib3
import numpy as np
import requests
output = panda_joke_cnn()
predict = tf.argmax(output, 1)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "./i_am_a_joke.model-100")



face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cnt=-1
font = cv2.FONT_HERSHEY_PLAIN
array_paint=[]


# **************************
host = "192.168.99.128:8081"
userName = "admin"
passWord  = "admin"
hoststr = 'http://'+userName+":"+passWord+'@' + host + '/video'
# stream=urllib3.urlopen(hoststr)
http = urllib3.PoolManager()
response = requests.get(hoststr, stream=True)
bytes=bytes()
while True:
    bytes+=response.raw.read(1024)
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')
    if a!=-1 and b!=-1:
        jpg = bytes[a:b+2]
        bytes= bytes[b+2:]
        img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
        cnt += 1
        print("cnt=",cnt)
        if cnt % 3 == 0:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_haar.detectMultiScale(gray_image, 1.3, 5)
            next_array_paint = []
            for face_x, face_y, face_w, face_h in faces:
                face = img[face_y:face_y + face_h, face_x:face_x + face_w]

                face = cv2.resize(face, (64, 64))

                # print(is_my_face(face))


                if is_my_face( face,predict,sess):
                    tag = "Shawn"
                    color = (255, 255, 255)
                else:
                    tag = 'Other'
                    color = (0, 255, 255)
                o1, o2 = (face_x, face_y), (face_x + face_w, face_y + face_h)
                next_array_paint.append([o1, o2, color])
                # cv2.rectangle(img, (face_x, face_y), (face_x + face_w, face_y + face_h), color, 2)
        if len(next_array_paint) == 0:
            if len(array_paint) == 0:
                array_paint = next_array_paint
        else:
            array_paint = next_array_paint
        for (p1, p2, cr) in array_paint:
            cv2.rectangle(img, p1, p2, cr, 2)
        cv2.imshow('img', img)
        array_paint = next_array_paint
        if cv2.waitKey(1) ==27:
            exit(0)
sess.close()


#
#
# while True:
#     _, img = cam.read()
#     cnt+=1
#     if cnt %3==0:
#         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_haar.detectMultiScale(gray_image, 1.3, 5)
#         next_array_paint=[]
#         for face_x, face_y, face_w, face_h in faces:
#             face = img[face_y:face_y + face_h, face_x:face_x + face_w]
#
#             face = cv2.resize(face, (64, 64))
#
#             # print(is_my_face(face))
#
#
#             if is_my_face(face):
#                 tag="Shawn"
#                 color=(255,0,0)
#             else:
#                 tag='Other'
#                 color=(0,255,255)
#             o1,o2=(face_x, face_y), (face_x + face_w, face_y + face_h)
#             next_array_paint.append([o1,o2,color])
#             # cv2.rectangle(img, (face_x, face_y), (face_x + face_w, face_y + face_h), color, 2)
#     if len(next_array_paint)==0:
#         if len(array_paint)==0:
#             array_paint=next_array_paint
#     else:
#         array_paint=next_array_paint
#     for (p1,p2,cr) in array_paint:
#         cv2.rectangle(img,p1,p2,cr,2)
#     cv2.imshow('img', img)
#     array_paint=next_array_paint
#     key = cv2.waitKey(30) & 0xff
#     if key == 27:
#         sys.exit(0)
