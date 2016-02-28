import numpy as np
import cv2

MIN_MATCH_COUNT = 10

img1 = cv2.imread('pos-82.pgm',0)
img2 = cv2.imread('car.jpg',0)
w, h = img1.shape[::-1]

sift = cv2.xfeatures2d.SURF_create(2000)

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
print(kp1)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)


print (matches)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for meth in methods:
  img = img2.copy()
  method = eval(meth)
# Apply template Matching
res = cv2.matchTemplate(des1,des2,cv2.TM_CCORR_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print (max_loc)
# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
  top_left = min_loc
else:
   top_left = max_loc


bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img2,top_left, bottom_right, 255, 2)


img2 = cv2.drawKeypoints(img2, kp2, np.array([]), (0,0,255), 2)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
cv2.imwrite('homography.jpg',img3)
