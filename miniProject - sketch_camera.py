import numpy
import cv2

def sketch(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img,60,100)
    ret,mask = cv2.threshold(canny,70,255,cv2.THRESH_BINARY)
    return mask



cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    frame = sketch(frame)
    cv2.imshow("Sketch webcam",frame)
    
    if cv2.waitKey(3) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("Camera disconnected")
