import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image,lineParameters):
    slope,intercept=lineParameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def avgSlopeDetection(image,lines):
    leftFit=[]
    rightFit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        if slope < 0:
            leftFit.append((slope,intercept))
        else:
            rightFit.append((slope,intercept))
    leftFitAvg=np.average(leftFit,axis=0)
    rightFitAvg=np.average(rightFit,axis=0)
    leftLine=make_coordinates(image,leftFitAvg)
    rightLine=make_coordinates(image,rightFitAvg)
    return np.array([leftLine,rightLine])
    # print(leftFitAvg)
    # print(rightFitAvg)

def canny(image):
    gray=cv2.cvtColor(laneImage,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny
def displayLine(image,lines):
    lineImage=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(lineImage,(x1,y1),(x2,y2),(255,0,0),10)
    return lineImage

def regionOfInterest(image):
    height=image.shape[0]
    polygons = np.array([
    [(200,height),(1100,height),(550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    maskedImage=cv2.bitwise_and(image,mask)
    return maskedImage

image=cv2.imread('test_image.jpg')  #convert image into numpy array
laneImage=np.copy(image)
# canny_image=canny(laneImage)
# croppedImage = regionOfInterest(canny_image)
# lines = cv2.HoughLinesP(croppedImage,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
# avgLines=avgSlopeDetection(laneImage,lines)
# lineImage=displayLine(laneImage,avgLines)
# comboImage=cv2.addWeighted(laneImage,0.8,lineImage,1,1)
# # plt.imshow(canny)
# # plt.show()
# cv2.imshow('results',comboImage)
# cv2.waitKey(0)

cap=cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _,frame=cap.read()
    canny_image=canny(frame)
    croppedImage = regionOfInterest(canny_image)
    lines = cv2.HoughLinesP(croppedImage,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    avgLines=avgSlopeDetection(frame,lines)
    lineImage=displayLine(frame,avgLines)
    comboImage=cv2.addWeighted(frame,0.8,lineImage,1,1)
    cv2.imshow('results',comboImage)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
