import numpy as np
import cv2
from time import clock
import threading
from time import ctime,sleep


lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(
    maxCorners = 500,
    qualityLevel = 0.3,
    minDistance = 7,
    blockSize = 7)
cascade_path = "haarcascade_frontalface_default.xml"
cap = cv2.VideoCapture(0)
faceFlag = False
rectGlobal = []
class LK:
    def __init__(self,vedio_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cap
        self.frame_idx = 0
        self.face = False
    def run(self):
        while True:
            if len(rectGlobal)<1:continue
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
                            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
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
                        rectGlobal[0][0] = rectGlobal[0][0] - int(diff_x / count)
                        rectGlobal[0][1] = rectGlobal[0][1] - int(diff_y / count)
                        rectGlobal[0][2] = rectGlobal[0][2] - int(diff_x / count)
                        rectGlobal[0][3] = rectGlobal[0][3] - int(diff_y / count)
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

def getRectangle(frame):
    cascade = cv2.CascadeClassifier(cascade_path)  # face
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))  # face
    color = (255, 255, 255)
    rect = []
    for r in facerect:
        rect.append(r)
    return rect
def LKrun():
    LK(0).run()

if __name__ == "__main__":
    ind = -1
    t = threading.Thread(target=LKrun)
    t.setDaemon(True)
    t.start()
    while True:
        ind = ind +1
        _, frame = cap.read()
        if ind % 50 == 0:
            if ind > 50000: ind = 0
            rectGlobal = getRectangle(frame)
            while len(rectGlobal)>1:
                rectGlobal = getRectangle(frame)
        for r in rectGlobal:
            color = (255, 255, 255)
            cv2.rectangle(frame, tuple(r[0:2]), tuple(r[0:2] + r[2:4]), color, thickness=2)
        cv2.imshow("win", frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break