import numpy as np
import cv2

names_file = open('carlist.txt', 'r')
names = names_file.readlines()
names_file.close()


img1 = cv2.imread(names[3],0)
img2 = cv2.imread('test-0.pgm',0)
w, h = img1.shape[::-1]
surf = cv2.xfeatures2d.SURF_create(300)

kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for meth in methods:
  img = img2.copy()
  method = eval(meth)
# Apply template Matching
res = cv2.matchTemplate(des2,des1,cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
  top_left = min_loc
else:
   top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img2,top_left, bottom_right, 255, 2)
cv2.imwrite("imatge.jpg",img2)


