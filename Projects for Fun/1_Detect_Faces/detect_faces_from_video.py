import numpy as np
import argparse
import cv2

prototxt='deploy.prototxt.txt'
model='res10_300x300_ssd_iter_140000.caffemodel'
conf=0.5

print('[INFO] Loading model...')
net=cv2.dnn.readNetFromCaffe(prototxt,model)

cam=cv2.VideoCapture(0)

def drawRects(img,detcts):
    image=img
    detections=detcts
    (h,w)=image.shape[:2]
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]      
        # Extracting confidence
        if confidence>conf:
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])   
            # Compute (x,y) coordinates for the bounding box
            (startX,startY,endX,endY)=box.astype('int')
            # Drawing bounding box and puting confidence text
            text='{:.2f}%'.format(confidence*100)
            y=startY-10 if startY-10>10 else startY+10
            cv2.rectangle(image,(startX,startY),(endX,endY),(0,0,255),2)
            cv2.putText(image,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
    cv2.imshow('Output',image)


while True:
    _,feed=cam.read()
    image=feed
    # blob = cv2.dnn.blobFromImages(images, scalefactor=1.0, size, mean, swapRB=True)
    blob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0,177.0,123.0)) #preprocessing Image
    net.setInput(blob)
    detections=net.forward()
    drawRects(image,detections)   
    if cv2.waitKey(1) & 0XFF==27:
        break
cam.release()
cv2.destroyAllWindows()