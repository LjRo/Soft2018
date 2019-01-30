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
        if(self.x <= 25 or self.y <= 25):
            return True
        else:
            return False 
    def distanceFrom(self,pos):
        return math.sqrt((self.x - pos.x)**2 + (self.y - pos.y)**2)

class HistoryUnit:
    def __init__(self, position, cFrame):
        self.position = position
        self.cFrame = cFrame

class Tracked:
    def __init__(self,cnt,position,w,h,value,number):
        self.cnt = cnt
        self.position = position
        self.w = w
        self.h = h
        self.value = value
        self.history = []
        self.number = number
    
    def updatePosition(self,cnt,position,w,h,value,fCount):
        if(self.cnt.shape == cnt.shape):
            if(self.position.isNear(position)):
                self.history.append(HistoryUnit(self.position,fCount))
                self.position = position
                return True
        else:
            if(self.position.isNear(position)):
                if((self.w - 1) <= w) and ((self.h - 1) <= h) and ((self.w + 1) >= w) and ((self.h + 1) >= h):
                    if(self.value == value):
                        self.history.append(HistoryUnit(self.position,fCount))
                        self.position = position
                        return True
        return False

    def avgSpeed(self):
        if(len(self.history) > 1):
            sPos = self.history[0].position
            ePos = self.history[-1].position

            distance = sPos.distanceFrom(ePos)
            frameCount = self.history[-1].cFrame - self.history[0].cFrame

            return distance/frameCount

        else:
            return 0

f= open("out.txt","w+")
f.write("RA 73/2015 Ljubomir Rokvic\r")
f.write("file	sum\r")
k=0
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

    fCount = 0
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
        
        cntCount = 0
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
                if((w > 1 and h > 14) or (w>14 and h>5)) and (w<=28 and h<=28) and hier[0][cntCount][3] == -1:
                    number = format(cnt,thresh[:,:])
                    ynew = model.predict_classes(number.reshape(1,28,28,1))
                    M = cv2.moments(cnt)
                    pos = Position(int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    nTr = Tracked(cnt,pos,w,h,ynew,number)
                    #nTr = Tracked(cnt,Position(x+int(w/2),y+int(h/2)),w,h,ynew,number)
                    globalTracked.append(nTr)
                cntCount+=1


        cntCount = 0
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            if((w > 1 and h > 14) or (w>14 and h>5)) and (w<=28 and h<=28) and hier[0][cntCount][3] == -1:
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                number = format(cnt,thresh[:,:])
                ynew = model.predict_classes(number.reshape(1,28,28,1))
                M = cv2.moments(cnt)
                pos = Position(int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                nTr = Tracked(cnt,pos,w,h,ynew,number)
                # nTr = Tracked(cnt,Position(x+int(w/2),y+int(h/2)),w,h,ynew,number)

                updated = False

                for tr in globalTracked:
                    #if(tr.updatePosition(cnt,Position(x+int(w/2),y+int(h/2)),w,h,ynew)):
                    if(tr.updatePosition(cnt,pos,w,h,ynew,fCount)):
                        updated = True
                        break
                if(not(updated)):
                    #print(str(x) + " - " + str(y))
                    if(nTr.position.isNearTLBorder()):
                        globalTracked.append(nTr)

                #cv2.imshow('Preview', number)
            cntCount += 1


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
        #Frame counter
        fCount +=1

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
        if(len(tr.history) > 13): 
            for i in range(0,len(tr.history)-1):
                cv2.line(lastframe,(tr.history[i].position.x,tr.history[i].position.y),(tr.history[i+1].position.x,tr.history[i+1].position.y),(255,0,0),1)
                if(intersect(tr.history[i].position,tr.history[i+1].position,bPoint1,bPoint2)):
                    countB += 1
                    rez += tr.value
                    #cv2.imwrite('debug/Blue_'+ str(randint(0,10000000))+'_' + str(tr.value[0]) +'.png',tr.number)
                    #print(tr.value)
                if(intersect(tr.history[i].position,tr.history[i+1].position,gPoint1,gPoint2)):
                    countG += 1
                    rez -= tr.value
                    #cv2.imwrite('debug/Green_'+ str(randint(0,10000000))+'_' + str(tr.value[0]) +'.png',tr.number)
                    #print(-tr.value)
            if(tr.history[-1].cFrame < (fCount-10)) and (tr.history[-1].position.x < 600) and (tr.history[-1].position.y < 440):
                sPoint = tr.history[-1].position
                xList = [history.position.x for history in tr.history[:]] 
                yList = [history.position.y for history in tr.history[:]]
                a,n = np.polyfit(xList,yList,1)
                #print(str(tr.value) +"K: " + str(a) + " - N:" + str(n) )
                
                theta = math.atan(a)

                aSpeed = tr.avgSpeed()
                length = aSpeed * (fCount - tr.history[-1].cFrame) 
                ePoint = Position(int(length*math.cos(theta)),int(length*math.sin(theta)))

                if(ePoint.x > 480):
                    ePoint.x = 480

                ePoint.y = int(ePoint.x * a + n)
                if(ePoint.y > 640):
                    ePoint.y = 640
                    ePoint.x = int((ePoint.y - n)/a) 
                cv2.line(lastframe,(sPoint.x,sPoint.y),(ePoint.x,ePoint.y),(255,255,0),1)

                if(intersect(sPoint,ePoint,bPoint1,bPoint2)):
                    countB += 1
                    rez += tr.value
                    #cv2.imwrite('debug/Blue_'+ str(randint(0,10000000))+'_' + str(tr.value[0]) +'.png',tr.number)
                    #print(tr.value)
                if(intersect(sPoint,ePoint,gPoint1,gPoint2)):
                    countG += 1
                    rez -= tr.value
                    #cv2.imwrite('debug/Green_'+ str(randint(0,10000000))+'_' + str(tr.value[0]) +'.png',tr.number)
                    #print(-tr.value)

    print("RESULT: " + str(rez))
    cv2.imshow('Preview', lastframe)
    #print(str(countB) + ' : ' + str(countG))
    f.write('video-'+str(k)+'.avi\t' + str(rez[0])+'\r')
    cv2.waitKey(5000)
    cap.release()
    cv2.destroyAllWindows() 

f.close()
