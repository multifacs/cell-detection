# import the necessary packages
import cv2
import numpy as np

IMAGE: str = "./images/image1.jpg"
CONNECTIVITY: int = 4

# load the input image from disk, convert it to grayscale, and
# threshold it
image = cv2.imread(IMAGE)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

# apply connected component analysis to the thresholded image
output = cv2.connectedComponentsWithStats(
	thresh, CONNECTIVITY, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

# initialize an output mask to store all characters parsed from
# the license plate
mask = np.zeros(gray.shape, dtype="uint8")

# loop over the number of unique connected component labels, skipping
# over the first label (as label zero is the background)
for i in range(1, numLabels):
	# extract the connected component statistics for the current
	# label
	x = stats[i, cv2.CC_STAT_LEFT]
	y = stats[i, cv2.CC_STAT_TOP]
	w = stats[i, cv2.CC_STAT_WIDTH]
	h = stats[i, cv2.CC_STAT_HEIGHT]
	area = stats[i, cv2.CC_STAT_AREA]
	(cX, cY) = centroids[i]
	
    # ensure the width, height, and area are all neither too small
	# nor too big
	keepWidth = w > 5 and w < 100
	keepHeight = h > 5 and h < 100
	keepArea = area > 50 and area < 1000
	# ensure the connected component we are examining passes all
	# three tests
	if all((keepWidth, keepHeight, keepArea)):
		# construct a mask for the current connected component and
		# then take the bitwise OR with the mask
		print("[INFO] keeping connected component '{}'".format(i))
		componentMask = (labels == i).astype("uint8") * 255
		mask = cv2.bitwise_or(mask, componentMask)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
		cv2.circle(image, (int(cX), int(cY)), 4, (0, 0, 255), -1)
		
# show the original input image and the mask for the license plate
# characters
cv2.imshow("Image", image)
cv2.imshow("Characters", mask)
cv2.waitKey(0)