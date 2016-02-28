import numpy as np
import cv2

img1 = cv2.imread('pos-82.pgm',0)
img2 = cv2.imread('test-0.pgm',0)

surf = cv2.xfeatures2d.SURF_create(400)

kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

print (matches)


# Copy data des1 and kp1 to a .txt file.
#Nota: S'hauria de mirar com ho llegeix per tal dafegir les caracteristiques de cada imatge 
#	a la base de dades.
#Es poden provar varis tipus de descriptors aixi mirem quin va millor per a cotxes.

datades = open('des.txt', 'w')

# Loop through each item in the list
# and write it to the output des file.
for eachitem in des1:
    datades.write(str(eachitem)+'\n')

# Close the output des file
datades.close()

# Open the file for writing
datakp = open('kp.txt', 'w')

# Loop through each item in the list
# and write it to the output kp file.
for eachitem in kp1:
    datakp.write(str(eachitem)+'\n')

# Close the output kp file
datakp.close()

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
cv2.imwrite("test.jpg",img3)


