#Import packages
import cv2 as cv
import numpy as np
import pyvisgraph as vg

def convertToSceneCoord( imageX, imageY, img_height, img_width ):
    w = 25
    h = 25

    sceneX = -( (imageY*h/img_height) - h/2 )
    sceneY = -( (imageX*w/img_width) - w/2 )

    return (sceneX, sceneY)

def convertFromSceneCoord( sceneX, sceneY, img_height, img_width ):
    w = 25.0
    h = 25.0

    imageY = int( ((-sceneX)+h/2)*img_height/h )
    imageX = int( ((-sceneY)+w/2)*img_width/w )

    return (imageX, imageY)

def getWallsLines(img):
    
    lower_color_wall = np.array([100,100,100])
    upper_color_wall = np.array([136,143,146])
    mask_wall = cv.inRange(img, lower_color_wall, upper_color_wall) 
    lines_wall = cv.HoughLinesP(mask_wall, 1, np.pi / 180, 3, np.array([]), 100, 50)

    lower_color_black = np.array([0,0,0])
    upper_color_black = np.array([2,2,2])
    mask_black = cv.inRange(img, lower_color_black, upper_color_black)
    lines_black = cv.HoughLinesP(mask_black, 1, np.pi / 180, 7, np.array([]), 30, 40)

    lines = np.append( lines_wall, lines_black, axis=0 )

    return lines

def generateRectangles(img, lines):

    rect_image = np.copy(img) * 0 # creating a blank to draw rectangles on

    scale=5
    rects=[]
    for line in lines:
        x1,y1,x2,y2 = line.ravel()
        v1 = np.array( [[x1, y1]] )
        v2 = np.array( [[x2, y2]] )

        orth = np.array( [[y1-y2, x2-x1]] )
        orth = orth/np.linalg.norm(orth)

        p1 = (v1+scale*orth).astype(int)
        p2 = (v1-scale*orth).astype(int)
        p3 = (v2+scale*orth).astype(int)
        p4 = (v2-scale*orth).astype(int)

        rect = [p1,p2,p4,p3]
        rects.append(rect)

        cv.rectangle(rect_image, (p1[0,0], p1[0,1]), (p4[0,0], p4[0,1]), (255,255,255))
        # cv.rectangle(rect_image, (p1[0,0], p1[0,1]), (p4[0,0], p4[0,1]), (255,255,255), -1) # filled rectangles

    rects = np.array(rects).reshape(-1, 4, 2)

    print(rects)

    return np.copy(rect_image), rects

##############

IMAGE_NAME = './image.jpeg'
img = cv.imread(IMAGE_NAME, cv.IMREAD_COLOR)

lines = getWallsLines(img)
rect_image, rects = generateRectangles(img, lines)

print("Number of rectangles: ", rects.shape[0])

cv.imshow('Original image', cv.resize(img, dsize=None, fx=0.7, fy=0.7)) 
cv.imshow('Extracted rectangles', cv.resize(rect_image, dsize=None, fx=0.7, fy=0.7))

# Create a list of polygons
polygons = []
for rect in rects:
    polygon = []
    for point in rect:
        polygon.append( vg.Point(point[0], point[1]) )
    polygons.append(polygon)

# Start building the visibility graph 
graph = vg.VisGraph()
print('Starting building visibility graph')
graph.build(polygons)
print('Finished building visibility graph')


cv.waitKey(0)
cv.destroyAllWindows()