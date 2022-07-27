# só a ideia, tem que arrumar a recursão

#Import packages
import cv2 as cv
from nbformat import convert
import numpy as np

def convertToSceneCoord( imageX, imageY, img_height, img_width ):
    w = 25
    h = 25

    sceneX = -( (imageY*h/img_height) - h/2 )
    sceneY = -( (imageX*w/img_width) - w/2 )

    return np.array( [sceneX, sceneY] )

def convertFromSceneCoord( sceneX, sceneY, img_height, img_width ):
    w = 25.0
    h = 25.0

    imageY = int( ((-sceneX)+h/2)*img_height/h )
    imageX = int( ((-sceneY)+w/2)*img_width/w )

    return np.array( [imageX, imageY] )

def getImageWalls(img):

    wall_image = np.copy(img) * 0  # creating a blank to draw lines on

    lower_color_wall = np.array([100,100,100])
    upper_color_wall = np.array([136,143,146])
    mask_wall = cv.inRange(img, lower_color_wall, upper_color_wall)
    lines_wall = cv.HoughLinesP(mask_wall, 1, np.pi / 180, 3, np.array([]), 100, 50)

    lower_color_black = np.array([0,0,0])
    upper_color_black = np.array([2,2,2])
    mask_black = cv.inRange(img, lower_color_black, upper_color_black)
    lines_black = cv.HoughLinesP(mask_black, 1, np.pi / 180, 7, np.array([]), 30, 40)

    lines = np.append( lines_wall, lines_black, axis=0 )

    for line in lines:
        x1,y1,x2,y2 = line.ravel()
        cv.line(wall_image,(x1,y1),(x2,y2),(255,255,255),10)

    return np.array(lines), wall_image

def getCorners(img):
    corners_image = np.copy(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, 50, 0.001, 5)
    corners = np.int0(corners)

    # print(corners)

    for corner in corners:
        x,y = corner.ravel()
        cv.circle(corners_image, (x,y), 5, (255,0,0), -1)

    return corners, corners_image

def checkIntersection(A, B, lines):
    # segments [A, B], [C, D]   

    intersects = False
    for line in lines:
        C = np.array( line[0:2] )
        D = np.array( line[2:4] )

        p = C-A
        q = B-A
        r = D-C

        t = (q[1]*p[0] - q[0]*p[1])/(q[0]*r[1] - q[1]*r[0]) \
            if (q[0]*r[1] - q[1]*r[0]) != 0 \
            else (q[1]*p[0] - q[0]*p[1])
        u = (p[0] + t*r[0])/q[0] \
            if q[0] != 0 \
            else (p[1] + t*r[1])/q[1]

        intersects = (t >= 0 and t <= 1 and u >= 0 and u <= 1)
        if intersects:
            break

    return intersects
        
    
def findPath( path, destination, corners, lines ):

    if sum( path[-1] != destination ) == 0:
        return True

    delta = corners - destination
    dist = np.linalg.norm(delta, axis=1)
    ordered_dist = np.argsort(dist)

    cur_point = path[-1]

    intersects=True
    for idx in ordered_dist:
        next_point = corners[idx]
        intersects = checkIntersection( cur_point, next_point, lines )
        if not intersects:
            new_path = path.copy()
            new_path.append( corners[idx] )
            new_corners = np.delete(corners, idx, axis=0)
            flag = findPath( new_path, destination, new_corners, lines )
            if flag:
                return True
        intersects=True

    return False

###

IMAGE_NAME = './image.jpeg'
img = cv.imread(IMAGE_NAME, cv.IMREAD_COLOR)

lines, lines_image = getImageWalls(img)
corners, corners_image = getCorners(lines_image)

origin = convertFromSceneCoord( -9.8195, -0.40, img.shape[0], img.shape[1])
destination = convertFromSceneCoord( +11.625, -11.65, img.shape[0], img.shape[1])

lines = lines.reshape( -1, 4 )
corners = corners.reshape( -1, 2 )

save_corners = np.copy( corners )

corners = np.append( corners, np.array( [destination] ), axis=0 )

path = [origin]

findPath( path.copy(), destination, np.copy(corners), lines )


cv.imshow('img',cv.resize(img, dsize=None, fx=0.7, fy=0.7)) 
cv.imshow('teste',cv.resize(lines_image, dsize=None, fx=0.7, fy=0.7))
cv.imshow('corners',cv.resize(corners_image, dsize=None, fx=0.7, fy=0.7)) 

cv.waitKey(0)
cv.destroyAllWindows()
