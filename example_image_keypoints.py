import cv2
from imread_from_url import imread_from_url

from dedodev2 import dedode_detector_L
from dedodev2.utils import get_best_device, draw_kpts


device = get_best_device()
detector = dedode_detector_L()

img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/L%C3%B6wenbrunnen_im_L%C3%B6wenhof.jpg/1280px-L%C3%B6wenbrunnen_im_L%C3%B6wenhof.jpg")
out = detector.detect(img, num_keypoints = 1024)

kps = out["keypoints"]
kps = detector.to_pixel_coords(kps, img.shape[0], img.shape[1])
keypoint_img = draw_kpts(img, kps[0])

cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)
cv2.imshow("Keypoints", keypoint_img)
cv2.waitKey(0)
