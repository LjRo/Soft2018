import cv2
import numpy as np
import random
import math
import keras
from random import randint
from keras.models import model_from_json

def format(cnt, image):
    x,y,w,h = cv2.boundingRect(cnt)
    target = image[y:(y+h),x:(x+w)]
    w_buffer = 0
    h_buffer = 0
    if w <28:
        w_buffer = int((28-w)/2)
    if h < 28:
        h_buffer = int((28-h)/2)
    target= cv2.copyMakeBorder(target,h_buffer,h_buffer+((28-h)%2),w_buffer,w_buffer+((28-w)%2),cv2.BORDER_CONSTANT,value=[0,0,0])

    return target

def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

class Position:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def isNear(self,pos):
        #threshhold = 27 Best so far , old 22
        threshhold = 27
        
        res = math.sqrt((self.x - pos.x)**2 + (self.y - pos.y)**2)
        if(res < threshhold):
            return True
        else:
            return False
    def isNearTLBorder(self):
        if(self.x <= 20 or self.y <= 20):
            return True
        else:
            return False 

class Tracked:
    def __init__(self,cnt,position,w,h,value,number):
        self.cnt = cnt
        self.position = position
        self.w = w
        self.h = h
        self.value = value
        self.history = []
        self.number = number
    
    def updatePosition(self,cnt,position,w,h,value):
        if(self.cnt.shape == cnt.shape):
            if(self.position.isNear(position)):
                self.history.append(self.position)
                self.position = position
                return True
        else:
            if(self.position.isNear(position)):
                if((self.w - 1) <= w) and ((self.h - 1) <= h) and ((self.w + 1) >= w) and ((self.h + 1) >= h):
                    if(self.value == value):
                        self.history.append(self.position)
                        self.position = position
                        return True
        return False


f= open("out.txt","w+")
f.write("RA 73/2015 Ljubomir Rokvic\r")
f.write("file	sum\r")
#k=0
for k in range(0,10):
    cap = cv2.VideoCapture('data/video-'+str(k)+'.avi')
    count = 0
    linesBlue = []
    linesGreen = []
    globalTracked = []
    first = True

    json_file = open('model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("model.h5")

    if cap.isOpened():
        ret,firstFrame = cap.read()


    while cap.isOpened():
        ret,frame = cap.read()

        if frame is None:
            break

        kernel = np.ones((3,3),np.uint8)
        kernelRed = np.ones((1,1),np.uint8)

        blueFrame = frame[:,:,0]
        greenFrame = frame[:,:,1]
        redFrame = frame[:,:,2]

        bFE = cv2.erode(blueFrame,kernel,iterations = 1)
        blueFrame = cv2.dilate(bFE,kernelRed,iterations = 1)
        #blueFrame = bFE

        gFE = cv2.erode(greenFrame,kernel,iterations = 1)
        greenFrame = cv2.dilate(gFE,kernelRed,iterations = 1)
        #greenFrame = gFE

        rFE = cv2.erode(redFrame,kernelRed,iterations = 1)
        redFrame = cv2.dilate(rFE,kernel,iterations = 1)

        ret,thresh = cv2.threshold(redFrame,150,255,0)
        (gImage, cnts,hier) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        thresh = cv2.erode(thresh,kernel,iterations = 1)
        

        if(first):
            minLineLength = 300
            maxLineGap = 100
            first = False
            pBlue = cv2.Canny(blueFrame, threshold1=200, threshold2=300)
            pBlue = cv2.GaussianBlur(pBlue,(7,7),0)
            pGreen = cv2.Canny(greenFrame, threshold1=200, threshold2=300)
            pGreen = cv2.GaussianBlur(pGreen,(7,7),0)
            linesBlue = cv2.HoughLinesP(pBlue,1,np.pi/180,100,minLineLength,maxLineGap)
            linesGreen = cv2.HoughLinesP(pGreen,1,np.pi/180,100,minLineLength,maxLineGap)
            #Hotfix for the first elements
            for cnt in cnts:
                x,y,w,h = cv2.boundingRect(cnt)
                if((w > 1 and h > 15) or (w>17 and h>5)) and (w<=28 and h<=28):
                    number = format(cnt,thresh[:,:])
                    ynew = model.predict_classes(number.reshape(1,28,28,1))
                    nTr = Tracked(cnt,Position(x+int(w/2),y+int(h/2)),w,h,ynew,number)
                    globalTracked.append(nTr)

        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            if((w > 1 and h > 15) or (w>17 and h>5)) and (w<=28 and h<=28):
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                number = format(cnt,thresh[:,:])
                ynew = model.predict_classes(number.reshape(1,28,28,1))
                nTr = Tracked(cnt,Position(x+int(w/2),y+int(h/2)),w,h,ynew,number)
                updated = False

                for tr in globalTracked:
                    if(tr.updatePosition(cnt,Position(x+int(w/2),y+int(h/2)),w,h,ynew)):
                        updated = True
                        break
                if(not(updated)):
                    #print(str(x) + " - " + str(y))
                    if(nTr.position.isNearTLBorder()):
                        globalTracked.append(nTr)

                #cv2.imshow('Preview', number)


        for x1,y1,x2,y2 in linesBlue[0]:
            cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

        for x1,y1,x2,y2 in linesGreen[0]:
            cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

        cv2.imshow('Preview', frame)
        #print(len(globalTracked))
        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        lastframe = frame

    countG = 0
    countB = 0
    rez = 0

    for x1,y1,x2,y2 in linesBlue[0]:
        bPoint1 = Position(x1,y1)
        bPoint2 = Position(x2,y2)

    for x1,y1,x2,y2 in linesGreen[0]:
        gPoint1 = Position(x1,y1)
        gPoint2 = Position(x2,y2)

    print(len(globalTracked))
    for tr in globalTracked:
        #print(len(tr.history))
        if(len(tr.history) > 50): 
            for i in range(0,len(tr.history)-1):
                cv2.line(lastframe,(tr.history[i].x,tr.history[i].y),(tr.history[i+1].x,tr.history[i+1].y),(255,0,0),1)
                if(intersect(tr.history[i],tr.history[i+1],bPoint1,bPoint2)):
                    countB += 1
                    rez += tr.value
                    #cv2.imwrite('debug/Blue_'+ str(randint(0,10000000))+'_' + str(tr.value[0]) +'.png',tr.number)
                    #print(tr.value)
                if(intersect(tr.history[i],tr.history[i+1],gPoint1,gPoint2)):
                    countG += 1
                    rez -= tr.value
                    #cv2.imwrite('debug/Green_'+ str(randint(0,10000000))+'_' + str(tr.value[0]) +'.png',tr.number)
                    #print(-tr.value)


    print("RESULT: " + str(rez))
    cv2.imshow('Preview', lastframe)
    #print(str(countB) + ' : ' + str(countG))
    f.write('video-'+str(k)+'.avi ' + str(rez[0])+'\r')
    cv2.waitKey(5000)
    cap.release()
    cv2.destroyAllWindows() 

f.close()
