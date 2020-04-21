import numpy as np
import cv2


def gaussianBlur(img):
    sigma = 1
    k = 3*sigma
    k_size = np.int(np.ceil((2*k+1)**2))
    gk = cv2.getGaussianKernel(k_size,sigma)
    blurred = cv2.filter2D(img,-1,gk)
    blurred = cv2.filter2D(blurred,-1,gk.T)
    return blurred

def compute_convex_hull(frame):
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	#gray = gaussianBlur(gray)
	canny = cv2.Canny(gray,70,160)
	_,contours,hierarchy = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

	try:
		main_contour = sorted(contours,key=cv2.contourArea,reverse=False)[-1]
		hull = cv2.convexHull(main_contour)
		cv2.drawContours(frame,[hull],-1,(0,255,0),2)
	except:
		print("sagne")
	return frame

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    hull = compute_convex_hull(frame)
    cv2.imshow("hull",hull)
    if cv2.waitKey(3) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()