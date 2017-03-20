# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.cross_validation import train_test_split
import random
import sys

my_image_path = 'my_faces'
others_image_path = 'other_people'

image_data = []
label_data = []
rectGloabl = []


def get_padding_size(image):
    h, w, _ = image.shape
    longest_edge = max(h, w)
    top, bottom, left, right = (0, 0, 0, 0)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    return top, bottom, left, right


def read_data(img_path, image_h=64, image_w=64):
    for filename in os.listdir(img_path):
        if filename.endswith('.jpg'):
            filepath = os.path.join(img_path, filename)
            image = cv2.imread(filepath)
            top, bottom, left, right = get_padding_size(image)
            image_pad = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image = cv2.resize(image_pad, (image_h, image_w))

            image_data.append(image)
            label_data.append(img_path)


print "1"
read_data(others_image_path)
read_data(my_image_path)
print "1"
image_data = np.array(image_data)
label_data = np.array([[0, 1] if label == 'my_faces' else [1, 0] for label in label_data])
print "1"
train_x, test_x, train_y, test_y = train_test_split(image_data, label_data, test_size=0.05,
                                                    random_state=random.randint(0, 100))
print "2"
# image (height=64 width=64 channel=3)
train_x = train_x.reshape(train_x.shape[0], 64, 64, 3)
test_x = test_x.reshape(test_x.shape[0], 64, 64, 3)

# nomalize
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0
print "3"
print(len(train_x), len(train_y))
print(len(test_x), len(test_y))

#############################################################
batch_size = 128
num_batch = len(train_x) // batch_size

X = tf.placeholder(tf.float32, [None, 64, 64, 3])  # 图片大小64x64 channel=3
Y = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)
print "4"


def panda_joke_cnn():
    W_c1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
    b_c1 = tf.Variable(tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, W_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob_5)

    W_c2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    b_c2 = tf.Variable(tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, W_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob_5)

    W_c3 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
    b_c3 = tf.Variable(tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, W_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob_5)

    # Fully connected layer
    W_d = tf.Variable(tf.random_normal([8 * 16 * 32, 512], stddev=0.01))
    b_d = tf.Variable(tf.random_normal([512]))
    dense = tf.reshape(conv3, [-1, W_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, W_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob_75)

    W_out = tf.Variable(tf.random_normal([512, 2], stddev=0.01))
    b_out = tf.Variable(tf.random_normal([2]))
    out = tf.add(tf.matmul(dense, W_out), b_out)
    return out


def train_cnn():
    output = panda_joke_cnn()

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())

        for e in range(50):
            for i in range(num_batch):
                batch_x = train_x[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y[i * batch_size: (i + 1) * batch_size]
                _, loss_, summary = sess.run([optimizer, loss, merged_summary_op],
                                             feed_dict={X: batch_x, Y: batch_y, keep_prob_5: 0.5, keep_prob_75: 0.75})

                summary_writer.add_summary(summary, e * num_batch + i)

                if (e * num_batch + i) % 100 == 0:
                    acc = accuracy.eval({X: test_x, Y: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                    print(e * num_batch + i, acc)
                    # save model
                    if acc > 0.98:
                        saver.save(sess, "./i_am_a_joke.model", global_step=e * num_batch + i)
                        sys.exit(0)


def is_my_face(image, predict, sess):
    res = sess.run(predict, feed_dict={X: [image / 255.0], keep_prob_5: 1.0, keep_prob_75: 1.0})
    if res[0] == 1:
        return True
    else:
        return False


def getRect(img):
    face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facerect = face_haar.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))  # face
    rect = []
    for r in facerect:
        rect.append(r)
    return rect


def doDetect():
    face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)
    cnt = -1
    font = cv2.FONT_HERSHEY_PLAIN
    array_paint = []
    while True:
        _, img = cam.read()
        cnt += 1
        if cnt % 3 == 0:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_haar.detectMultiScale(gray_image, 1.3, 5)
            next_array_paint = []
            for face_x, face_y, face_w, face_h in faces:
                face = img[face_y:face_y + face_h, face_x:face_x + face_w]

                face = cv2.resize(face, (64, 64))

                # print(is_my_face(face))


                if is_my_face(face, predict, sess):
                    tag = "Shawn"
                    color = (255, 0, 0)
                    print "yes, this is me!"
                else:
                    tag = 'Other'
                    color = (0, 0, 255)
                    print "who are you?"
                o1, o2 = (face_x, face_y), (face_x + face_w, face_y + face_h)
                next_array_paint.append([o1, o2, color])
                # cv2.rectanqgle(img, (face_x, face_y), (face_x + face_w, face_y + face_h), color, 2)
        if len(next_array_paint) == 0:
            if len(array_paint) == 0:
                array_paint = next_array_paint
        else:
            array_paint = next_array_paint
        for (p1, p2, cr) in array_paint:
            cv2.rectangle(img, p1, p2, cr, 2)
        cv2.imshow('img', img)
        array_paint = next_array_paint
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)


# print "jinru"
# train_cnn()
print "5"
output = panda_joke_cnn()
predict = tf.argmax(output, 1)
saver = tf.train.Saver()
sess = tf.Session()
ckpt = tf.train.get_checkpoint_state('./')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, "./i_am_a_joke.model-100")
    print("recovery")

doDetect()
sess.close()