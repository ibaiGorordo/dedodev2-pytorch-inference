from imread_from_url import imread_from_url

from dedodev2 import dedode_detector_L, dedode_descriptor_B
from dedodev2.dual_softmax_matcher import DualSoftMaxMatcher
from dedodev2.utils import get_best_device, draw_matches
import cv2


device = get_best_device()
detector = dedode_detector_L()
descriptor = dedode_descriptor_B()
matcher = DualSoftMaxMatcher()

img1 = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/L%C3%B6wenbrunnen_im_L%C3%B6wenhof.jpg/1280px-L%C3%B6wenbrunnen_im_L%C3%B6wenhof.jpg")
img2 = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/2016-07-19_Fountain_de_los_Leones%2C_Patio_de_los_Leones.JPG/1280px-2016-07-19_Fountain_de_los_Leones%2C_Patio_de_los_Leones.JPG")

detections_1 = detector.detect(img1, num_keypoints = 1024)
detections_2 = detector.detect(img2, num_keypoints = 1024)

keypoints_1 = detections_1["keypoints"]
keypoints_2 = detections_2["keypoints"]


description_1 = descriptor.describe_keypoints(img1, keypoints_1)["descriptions"]
description_2 = descriptor.describe_keypoints(img2, keypoints_2)["descriptions"]

threshold = 0.6#Increasing threshold -> fewer matches, fewer outliers
indices_1, indices_2, probabilities,  = matcher.match(keypoints_1, description_1,
    keypoints_2, description_2,
    normalize = True, inv_temp=20, threshold = threshold)

matched_keypoints_1 = keypoints_1[0, indices_1]
matched_keypoints_2 = keypoints_2[0, indices_2]

matched_keypoints_1, matched_keypoints_2 = matcher.to_pixel_coords(matched_keypoints_1, matched_keypoints_2, img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])

img = draw_matches(img1, matched_keypoints_1, img2, matched_keypoints_2)

cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
cv2.imshow("Matches", img)
cv2.imwrite("matches.jpg", img)
cv2.waitKey(0)