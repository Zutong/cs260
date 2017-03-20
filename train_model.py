# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.cross_validation import train_test_split
import random
import sys
import threading
glo = 0
my_image_path = 'my_faces'
others_image_path = 'other_people'
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(
    maxCorners = 500,
    qualityLevel = 0.3,
    minDistance = 7,
    blockSize = 7)
faceFlag = False
rectGlobal = []
image_data = []
label_data = []
cam = cv2.VideoCapture(0)


class LK:
    def __init__(self,vedio_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cam
        self.frame_idx = 0
        self.face = False
    def run(self):
        while True:
            print len(rectGlobal)
            try:
                if len(rectGlobal)<1:continue
            except:
                continue
            try:
                ret, frame = self.cam.read()
                x, y = rectGlobal[0][0:2]
                w, h = rectGlobal[0][2:4]
                frame = frame[y - 10:y + h, x:x + w]
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if ret == True:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    vis = frame.copy()
                    if len(self.tracks) > 0:
                        img0, img1 = self.prev_gray, frame_gray
                        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                        good = d < 1
                        new_tracks = []
                        for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                            if not good_flag:
                                continue
                            tr.append((x, y))
                            if len(tr) > self.track_len:
                                del tr[0]
                            new_tracks.append(tr)
                            # cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                        self.tracks = new_tracks
                        '''-------------------------------------------------'''
                        good_new = p1[st == 1]
                        good_old = p0r[st == 1]
                        diff_x = 0
                        diff_y = 0
                        count = 0
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            diff_x = diff_x + (c - a)
                            diff_y = diff_y + (d - b)
                            count += 1
                        # print rectGlobal
                        rectGlobal[0][0] = rectGlobal[0][0] - 2*int(diff_x / count)
                        rectGlobal[0][1] = rectGlobal[0][1] - 2*int(diff_y / count)
                        rectGlobal[0][2] = rectGlobal[0][2] - 2*int(diff_x / count)
                        rectGlobal[0][3] = rectGlobal[0][3] - 2*int(diff_y / count)
                        # cv2.rectangle(vis, tuple(rr[0:2]), tuple(rr[0:2] + rr[2:4]), (0,255,0), thickness=2)
                        '''---------------------------------------------------'''
                        cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                    if self.frame_idx % self.detect_interval == 0:
                        mask = np.zeros_like(frame_gray)
                        mask[:] = 255
                        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                            cv2.circle(mask, (x, y), 5, 0, -1)
                        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                        if p is not None:
                            for x, y in np.float32(p).reshape(-1, 2):
                                self.tracks.append([(x, y)])
                    self.frame_idx += 1
                    self.prev_gray = frame_gray
                    # cv2.imshow('lk_track', vis)
                ch = 0xFF & cv2.waitKey(1)
                if ch == 27:
                    break

            except:

                    self.frame_idx += 1
                    self.prev_gray = frame_gray
                    continue

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


def getRectangle(frame):
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # face
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))  # face
    rect = []
    for r in facerect:
        rect.append(r)
    return rect


def LKrun():
    LK(0).run()




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

# print "jinru"
# train_cnn()
print "5"
output = panda_joke_cnn()
predict = tf.argmax(output, 1)
saver = tf.train.Saver()
sess = tf.Session()
ckpt = tf.train.get_checkpoint_state('./')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, "./i_am_a_joke.model-100" )
    print("recovery")
glo = 5
print glo

t = threading.Thread(target=LKrun)
t.setDaemon(True)
t.start()

cnt = -1
array_paint = []
while True:
    _, img = cam.read()
    cnt += 1
    if cnt % 50 == 0:
        rectGlobal = getRectangle(img)
        print "20yidao-----------------", len(rectGlobal)
        next_array_paint = []
        for face_x, face_y, face_w, face_h in rectGlobal:
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
    for r in rectGlobal:
        cv2.rectangle(img, tuple(r[0:2]), tuple(r[0:2] + r[2:4]), color, thickness=2)
    # for (p1, p2, cr) in array_paint:
    #     cv2.rectangle(img, p1, p2, cr, 2)
    cv2.imshow('img', img)
    array_paint = next_array_paint
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        sys.exit(0)

sess.close()