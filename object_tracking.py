import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

# cap = cv2.VideoCapture('E:\\AI\\images\\banli.mp4')
cap=cv2.VideoCapture(sys.argv[1])
ret, frame = cap.read()

h, w, _ = frame.shape

# 2/3
frame = cv2.resize(frame, (int(w * 2 / 3), int(h * 2 / 3)))
roix, roiy, roiw, roih = 80, 240, 210, 230
roi = frame[roiy:roiy + roih, roix:roix + roiw]
roi=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(roi, None)

# flann
FLANN_PARAM = 0
index_params = dict(algorithm=FLANN_PARAM, trees=5)
search_params = dict(checks=50)
flanner = cv2.FlannBasedMatcher(index_params, search_params)


# compute track_box
def compute_track(kp1, des1, kp2, des2, flanner, roi, match_rate, M_bef=None):
    matches = flanner.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < match_rate * n.distance: good.append(m)
    src_pts = np.array([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good])
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    matches = mask.ravel().tolist()
    h, w, _ = roi.shape
    track_box = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    try:
        track_box = cv2.perspectiveTransform(track_box, M)
    except Exception:
        M = M_bef
        track_box = cv2.perspectiveTransform(track_box, M)
        print('the match key points is less.')
    return good, matches, np.int32(track_box), M


count = 0

while (1):
    ret, frame = cap.read()
    if ret == True:
        count += 1
        h, w, _ = frame.shape
        # 2/3
        frame = cv2.resize(frame, (int(w * 2 / 3), int(h * 2 / 3)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kp2, des2 = sift.detectAndCompute(frame, None)

        good, matches, track_box, M_bef = compute_track(kp1, des1, kp2, des2, flanner, roi, 0.75, None)

        print('count:', count, 'good matches:', len(good))

        img_track = cv2.polylines(frame, [track_box], True, 255, 2)

        img_detection = cv2.drawMatches(roi, kp1, img_track, kp2, good, np.array([]), matchesMask=matches,
                                        matchColor=(0, 0, 255), singlePointColor=(255, 255, 0), flags=2)

        plt.imshow(img_detection)
        plt.pause(0.001)
        plt.clf()

        # if (len(good) > 2 * len(good_b)):
        #     img_track = cv2.polylines(frame, [track_box], True, 255, 2)
        #
        #     img_detection = cv2.drawMatches(roi, kp1, img_track, kp2, good, np.array([]), matchesMask=matches,
        #                                     matchColor=(0, 0, 255), singlePointColor=(255, 255, 0), flags=2)
        #     plt.imshow(img_detection)
        #     plt.pause(0.001)
        #     plt.clf()
        # else:
        #     # pass
        #     img_track = cv2.polylines(frame, [track_box_b], True, 255, 2)
        #
        #     img_detection = cv2.drawMatches(roi2, kp1_backup, img_track, kp2, good_b, np.array([]),
        #                                     matchesMask=matches_b,
        #                                     matchColor=(0, 0, 255), singlePointColor=(255, 255, 0), flags=2)
        #     plt.imshow(img_detection)
        #
        #     # plt.show()
        #     plt.pause(0.001)
        #     plt.clf()

        k = cv2.waitKey(1)
        if (k == 27):
            break
    else:
        break
