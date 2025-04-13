from imread_from_url import imread_from_url

from dedodev2 import dedode_detector_L, dedode_descriptor_B
from dedodev2.dual_softmax_matcher import DualSoftMaxMatcher
from dedodev2.utils import get_best_device, draw_matches
import cv2


# Read images from URLs
img1 = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/L%C3%B6wenbrunnen_im_L%C3%B6wenhof.jpg/1280px-L%C3%B6wenbrunnen_im_L%C3%B6wenhof.jpg")
img2 = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/2016-07-19_Fountain_de_los_Leones%2C_Patio_de_los_Leones.JPG/1280px-2016-07-19_Fountain_de_los_Leones%2C_Patio_de_los_Leones.JPG")

# Initialize the detector, descriptor, and matcher
detector = dedode_detector_L()
descriptor = dedode_descriptor_B()
matcher = DualSoftMaxMatcher()

# Detect keypoints
keypoints_1, _ = detector.detect(img1, num_keypoints = 1024)
keypoints_2, _ = detector.detect(img2, num_keypoints = 1024)

# Compute descriptors
descriptors_1 = descriptor.describe_keypoints(img1, keypoints_1)
descriptors_2 = descriptor.describe_keypoints(img2, keypoints_2)

# Match descriptor, increasing threshold -> fewer matches, fewer outliers
threshold = 0.6
match_indices_1, match_indices_2, probabilities = matcher.match(descriptors_1,
                                                                descriptors_2,
                                                                threshold = threshold)

# Filter out matches with low probability
matched_keypoints_1 = keypoints_1[match_indices_1]
matched_keypoints_2 = keypoints_2[match_indices_2]

# Draw matches
img = draw_matches(img1, matched_keypoints_1, img2, matched_keypoints_2)

cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
cv2.imshow("Matches", img)
cv2.waitKey(0)