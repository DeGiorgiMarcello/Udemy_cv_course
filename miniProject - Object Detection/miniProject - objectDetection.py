import numpy as np
import cv2

def sift_detector(image,template):

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = image.astype(np.uint8)
    sift = cv2.xfeatures2d.SIFT_create()
    
    image_keypoints,image_descriptor = sift.detectAndCompute(image,None)
    template_keypoints,template_descriptor = sift.detectAndCompute(template,None)

    #define parameters for the Flann Matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
    search_params = dict(checks = 100)

    #create the Flann object
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    #obtain matches using K-nearest Neighbor method
    matches = flann.knnMatch(image_descriptor,template_descriptor,k=2)

    #store good matches using Lowe's ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            
    return len(good_matches)



cap = cv2.VideoCapture(0)
_,frame = cap.read()
h,w = frame.shape[:2]
template = cv2.imread('obj_det_template.jpg',0)

#ROI
top_left_x = int(w/2) - 100
top_left_y = int(h/2) - 100
bottom_right_x = top_left_x + 200
bottom_right_y = top_left_y + 200

while cap.isOpened():
    ret,frame = cap.read()
    cv2.rectangle(frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(0,0,255),3)
    cropped = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    flip = cv2.flip(frame,1)
    matches = sift_detector(cropped,template)
    cv2.putText(frame,str(matches),(bottom_right_x+100,bottom_right_y +50),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
    
    #threshold for the number of found matches
    threshold = 10

    if matches > threshold:
        cv2.rectangle(frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(0,255,0),3) #turn red
        cv2.putText(frame,"Object_found",(int(w/2)-100,bottom_right_y+50),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

    cv2.imshow('Frame',frame)
    if cv2.waitKey(3) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    


    
                            
