import cv2  
import numpy as np;  
  
path = r"../CRBD/0-CRBD-bag/bag (1).jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  
cv2.imwrite("gs_1.jpg", img)

size = (320, 240) 
blob = cv2.dnn.blobFromImage(img,
                             scalefactor=1/255,
                             size=size,
                             swapRB=True)
 
# let's see our transformed image- blob
# print(blob)
cv2.imwrite("blob_1.jpg", blob)
print(f'Blob Shape : {np.array(blob).shape}')

""" # Set up the detector with default parameters.  
detector = cv2.SimpleBlobDetector()  

# Detecting blobs.  
keypoints = detector.detect(img)   """
print("BUrada")
# Draw detected blobs as red circles.  
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob  
""" im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),  
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  
# Show keypoints  
cv2.imwrite("Keypoints.jpg", im_with_keypoints)   """
cv2.waitKey(0)  