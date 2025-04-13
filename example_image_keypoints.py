import cv2
from imread_from_url import imread_from_url

from dedodev2 import dedode_detector_L
from dedodev2.utils import draw_kpts

detector = dedode_detector_L()

img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/L%C3%B6wenbrunnen_im_L%C3%B6wenhof.jpg/1280px-L%C3%B6wenbrunnen_im_L%C3%B6wenhof.jpg")
keypoints, confidences = detector.detect(img, num_keypoints = 1024)

keypoint_img = draw_kpts(img, keypoints)

cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)
cv2.imshow("Keypoints", keypoint_img)
cv2.waitKey(0)
