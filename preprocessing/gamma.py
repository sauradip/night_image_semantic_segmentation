from __future__ import print_function
import numpy as np
import argparse
import cv2


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

original = cv2.imread('/home/sauradip/Desktop/code/icip19/preprocessing/aachen_000000_000019_leftImg8bit.png')
gamma = 0.35
adjusted = adjust_gamma(original, gamma=gamma)
cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
cv2.imshow("Images", np.hstack([original, adjusted]))
cv2.waitKey(0)

    # loop over various values of gamma
# for gamma in np.arange(0.0, 1, 0.2):
# 	# ignore when gamma is 1 (there will be no change to the image)
# 	if gamma == 1:
# 		continue
# 	# apply gamma correction and show the images
# 	gamma = gamma if gamma > 0 else 0.1
	# adjusted = adjust_gamma(original, gamma=gamma)
	# cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
	# 	cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	# cv2.imshow("Images", np.hstack([original, adjusted]))
	# cv2.waitKey(0)
