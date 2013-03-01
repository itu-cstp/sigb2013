import cv2
import cv
import pylab
from SIGBTools import RegionProps
from SIGBTools import getLineCoordinates
from SIGBTools import ROISelector
from SIGBTools import getImageSequence
import numpy as np
import math
import sys

inputFile = "Sequences/eye1.avi"
outputFile = "eyeTrackerResult.mp4"

#--------------------------
#         Global variable
#--------------------------
global imgOrig,leftTemplate,rightTemplate,frameNr
imgOrig = [];
#These are used for template matching
leftTemplate = []
rightTemplate = []
frameNr =0

props = RegionProps()
def GetPupil(gray,thr,minArea=4200,maxArea=6000):
    """
    Doesn't work when eye is looking down. Be more loose with circularity
    """
    props = RegionProps()

    val,binI =cv2.threshold(gray, thr, 200, cv2.THRESH_BINARY_INV)
    
    binI = cv2.morphologyEx(binI,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS, (4,4)))

    # binI = cv2.morphologyEx(binI,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_CROSS,(4,4)))


    #Calculate blobs
    contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    match = []
    for con in contours:
        a = cv2.contourArea(con)
        # extend = props.CalcContourProperties(con,properties=["extend"]) # We don't use this because it's not needed

        if(a==0 or a<minArea or a>maxArea):
            continue
        p = cv2.arcLength(con, True)
        m = p/(2.0*math.sqrt(math.pi * a))
        if (m<2.9):
            ellips = cv2.fitEllipse(con)
            match.append(ellips)
    return match
    # xs = sorted(match, key=lambda x: cv2.contourArea(x),reverse=True)
    #
    # xs2 = []
    # for x in xs:
    #     if(len(x)>=5):
    #         cv2.drawContours(gray,[x],0,(255,0,0),1)
    #         xs2.append(cv2.fitEllipse(x))
    #
    # return xs2

def GetGlints(gray,thr,size):
        ''' Given a gray level image, gray and threshold
        value return a list of glint locations'''
        # YOUR IMPLEMENTATION HERE !!!!

        props = RegionProps()

        gray = gray.copy()
        M,N = gray.shape
        gray2 = np.zeros(gray.shape)

        val, binI = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
        # Opening
        binI = cv2.morphologyEx(binI,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(20,20)))
        contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


        match = []
        for con in contours:
            a = cv2.contourArea(con)

            if(a<240 and a>70):
                # cv2.drawContours(gray2,[con],0,(255,0,0),1)
                match.append(con)

        r = []
        for m in match:
            if(len(m)>=5):
                e = cv2.fitEllipse(m)
                r.append(e)


        # returning a list of candidate ellipsis
        return r

## Threshold
## Blob of proper size
## Blob of Shape

def Distance(a, b):
    """
    Calculates distance between two 2d points.

    """
    x1,y1 = a
    x2,y2 = b
    return math.sqrt(math.pow((x2-x1),2)+math.pow((y2-y1),2))


def GetIrisUsingThreshold(gray,pupil):
	''' Given a gray level image, gray and threshold
	value return a list of iris locations'''
	# YOUR IMPLEMENTATION HERE !!!!
	pass

def circularHough(gray):
	''' Performs a circular hough transform of the image, gray and shows the  detected circles
	The circe with most votes is shown in red and the rest in green colors '''
 #See help for http://opencv.itseez.com/modules/imgproc/doc/feature_detection.html?highlight=houghcircle#cv2.HoughCircles
	blur = cv2.GaussianBlur(gray, (31,31), 11)

	dp = 6; minDist = 30
	highThr = 20 #High threshold for canny
	accThr = 850; #accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected
	maxRadius = 50;
	minRadius = 155;
	circles = cv2.HoughCircles(blur,cv2.cv.CV_HOUGH_GRADIENT, dp,minDist, None, highThr,accThr,maxRadius, minRadius)

	#Make a color image from gray for display purposes
	gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
	if (circles !=None):
	 #print circles
	 all_circles = circles[0]
	 M,N = all_circles.shape
	 k=1
	 for c in all_circles:
			cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (int(k*255/M),k*128,0))
			K=k+1
	 c=all_circles[0,:]
	 cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (0,0,255),5)
	 cv2.imshow("hough",gColor)

def GetIrisUsingNormals(gray,pupil,normalLength):
	''' Given a gray level image, gray and the length of the normals, normalLength
	 return a list of iris locations'''
	# YOUR IMPLEMENTATION HERE !!!!
	pass

def GetIrisUsingSimplifyedHough(gray,pupil):
	''' Given a gray level image, gray
	return a list of iris locations using a simplified Hough transformation'''
	# YOUR IMPLEMENTATION HERE !!!!
	pass

def GetEyeCorners(leftTemplate, rightTemplate,pupilPosition=None):
	pass

def FilterPupilGlint(glints, pupils):
    glintList = [] #should be a set instead
    centerPoint = (0,0)
    for candA in glints:
        for candB in glints:
            #only accepting points with a certain distance to each other.
            if (Distance(candA[0],candB[0])> 45 and Distance(candA[0],candB[0]) < 55):
                glintList.append(candA)
                glintList.append(candB)
                centerPoint = (candA[0][0]+candB[0][0]/2, candA[0][1]+candB[0][1]/2)
    pupilList = []
    #sort out the pupils too far away from the center point between the latest found glints.
    for pupil in pupils:
        if (Distance(pupil[0], centerPoint) < 300):
            pupilList.append(pupil)
    return (glintList,pupilList)



# vwriter = cv2.VideoWriter("test.avi",('F','F','V','1'));
def update(I):
    '''Calculate the image features and display the result based on the slider values
    :param I:
    '''
    #global drawImg
    global frameNr,drawImg
    img = I.copy()
    sliderVals = getSliderVals()
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

# Do the magic  pupils = contour, glints = ellipse
    pupils = GetPupil(gray,sliderVals['pupilThr'],sliderVals['minSize'],sliderVals['maxSize'])
    glints = GetGlints(gray,sliderVals['glintThr'],100)
    glints, pupils = FilterPupilGlint(glints,pupils)


    for pupil in pupils:
        cv2.ellipse(img, pupil, (255,0,0),2)
    #    cv2.circle(img,(int(pupil[0][0]),int(pupil[0][1])),5,(0,255,0)) # Since we have an allipse we use it to find the center
    # for pupil in pupils:
    #    # cv2.ellipse(img,pupil,(0,255,0),2)
    #    C = int(pupil[0][0]),int(pupil[0][1])
    #    cv2.circle(img,C, 2, (0,0,255),4)

    for glint in glints:
        cv2.ellipse(img, glint,(0,255,0),2)


    #Do template matching
    global leftTemplate
    global rightTemplate
    GetEyeCorners(leftTemplate, rightTemplate)
    #Display results
    global frameNr,drawImg
    x,y = 10,10
    #setText(img,(x,y),"Frame:%d" %frameNr)
    sliderVals = getSliderVals()

    # for non-windows machines we print the values of the threshold in the original image
    if sys.platform != 'win32':
        step=18
    #    cv2.putText(img, "pupilThr :"+str(sliderVals['pupilThr']), (x, y+step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
    #    cv2.putText(img, "glintThr :"+str(sliderVals['glintThr']), (x, y+2*step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


		#Uncomment these lines as your methods start to work to display the result in the
		#original image

		#     cv2.imshow("Result", img)

		#For Iris detection - Week 2
		#circularHough(gray)

    #copy the image so that the result image (img) can be saved in the movie
    drawImg = img.copy()

    cv2.imshow('Result',img)

    # cv2.imshow('Temp',I)


def printUsage():
	print "Q or ESC: Stop"
	print "SPACE: Pause"
	print "r: reload video"
	print 'm: Mark region when the video has paused'
	print 's: toggle video  writing'
	print 'c: close video sequence'

def run(fileName,resultFile='eyeTrackingResults.avi'):

	''' MAIN Method to load the image sequence and handle user inputs'''
        
	global imgOrig, frameNr,drawImg;
	setupWindowSliders()
	props = RegionProps()
	cap,imgOrig,sequenceOK = getImageSequence(fileName)
	videoWriter = 0;

	frameNr =0
	if(sequenceOK):
		update(imgOrig)
	printUsage()
	frameNr=0;
	saveFrames = False
	while(sequenceOK):
		sliderVals = getSliderVals();
		frameNr=frameNr+1
		ch = cv2.waitKey(1)
		#Select regions
		if(ch==ord('m')):
			if(not sliderVals['Running']):
				roiSelect=ROISelector(imgOrig)
				pts,regionSelected= roiSelect.SelectArea('Select left eye corner',(400,200))
				if(regionSelected):
					leftTemplate = imgOrig[pts[0][1]:pts[1][1],pts[0][0]:pts[1][0]]

		if ch == 27:
			break
		if (ch==ord('s')):
			if((saveFrames)):
				videoWriter.release()
				saveFrames=False
				print "End recording"
			else:
				imSize = np.shape(imgOrig)
				videoWriter = cv2.VideoWriter(resultFile, cv.CV_FOURCC('D','I','V','3'), 15.0,(imSize[1],imSize[0]),True) #Make a video writer
				saveFrames = True
				print "Recording..."



		if(ch==ord('q')):
			break
		if(ch==32): #Spacebar
			sliderVals = getSliderVals()
			cv2.setTrackbarPos('Stop/Start','Threshold',not sliderVals['Running'])
		if(ch==ord('r')):
			frameNr =0
			sequenceOK=False
			cap,imgOrig,sequenceOK = getImageSequence(fileName)
			update(imgOrig)
			sequenceOK=True

		sliderVals=getSliderVals()
		if(sliderVals['Running']):
			sequenceOK, imgOrig = cap.read()
			if(sequenceOK): #if there is an image
				update(imgOrig)
			if(saveFrames):
				videoWriter.write(drawImg)


	# videoWriter.release



#--------------------------
#         UI related
#--------------------------

def setText(dst, (x, y), s):
	cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
	cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


def setupWindowSliders():
	''' Define windows for displaying the results and create trackbars'''
	cv2.namedWindow("Result")
	cv2.namedWindow('Threshold')
	#cv2.namedWindow("Temp")
        cv2.namedWindow("Aux")
	#Threshold value for the pupil intensity
	cv2.createTrackbar('pupilThr','Threshold', 108, 255, onSlidersChange)
	#Threshold value for the glint intensities
	cv2.createTrackbar('glintThr','Threshold', 240, 255,onSlidersChange)
	#define the minimum and maximum areas of the pupil
	cv2.createTrackbar('minSize','Threshold', 40, 200, onSlidersChange)
	cv2.createTrackbar('maxSize','Threshold', 120,200, onSlidersChange)
	#Value to indicate whether to run or pause the video
	cv2.createTrackbar('Stop/Start','Threshold', 0,1, onSlidersChange)

def getSliderVals():
	'''Extract the values of the sliders and return these in a dictionary'''
	sliderVals={}
	sliderVals['pupilThr'] = cv2.getTrackbarPos('pupilThr', 'Threshold')
	sliderVals['glintThr'] = cv2.getTrackbarPos('glintThr', 'Threshold')
	sliderVals['minSize'] = 50*cv2.getTrackbarPos('minSize', 'Threshold')
	sliderVals['maxSize'] = 50*cv2.getTrackbarPos('maxSize', 'Threshold')
	sliderVals['Running'] = 1==cv2.getTrackbarPos('Stop/Start', 'Threshold')
	return sliderVals

def onSlidersChange(dummy=None):
	''' Handle updates when slides have changed.
	 This  function only updates the display when the video is put on pause'''
	global imgOrig;
	sv=getSliderVals()
	if(not sv['Running']): # if pause
		update(imgOrig)

#--------------------------
#         main
#--------------------------
run(inputFile)

#img = cv2.imread("Sequences/eye.png")
# img = np.ones()
#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# print "y: " + y
#cv2.namedWindow("contour")
#cv2.imshow("contour", img)
#cv2.waitKey(0)

