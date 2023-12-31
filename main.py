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

# initialize an output mask to store all characters parsed from
# the license plate
mask = np.zeros(gray.shape, dtype="uint8")

print(thresh)

# apply connected component analysis to the thresholded image
output = cv2.connectedComponentsWithStats(
	thresh, CONNECTIVITY, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

output = image.copy()

# loop over the number of unique connected component labels
for i in range(0, numLabels):
	# if this is the first component then we examine the
	# *background* (typically we would just ignore this
	# component in our loop)
	if i == 0:
		text = "examining component {}/{} (background)".format(
			i + 1, numLabels)
		componentMask = (labels == i).astype("uint8") * 255
	# otherwise, we are examining an actual connected component
	else:
		text = "examining component {}/{}".format( i + 1, numLabels)
	# print a status message update for the current connected
	# component
	print("[INFO] {}".format(text))
	# extract the connected component statistics and centroid for
	# the current label
	x = stats[i, cv2.CC_STAT_LEFT]
	y = stats[i, cv2.CC_STAT_TOP]
	w = stats[i, cv2.CC_STAT_WIDTH]
	h = stats[i, cv2.CC_STAT_HEIGHT]
	area = stats[i, cv2.CC_STAT_AREA]
	(cX, cY) = centroids[i]
	
    # clone our original image (so we can draw on it) and then draw
	# a bounding box surrounding the connected component along with
	# a circle corresponding to the centroid
	
	cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
	cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
	
    # construct a mask for the current connected component by
	# finding a pixels in the labels array that have the current
	# connected component ID
	# componentMask = (labels == i).astype("uint8") * 255
	# show our output image and connected component mask

cv2.imshow("Output", output)
cv2.imshow("Connected Component", componentMask)
cv2.waitKey(0)