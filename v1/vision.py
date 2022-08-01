#Import packages
import cv2 as cv
import numpy as np

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

def getImageWalls(img):
    wall_image = np.copy(img) * 0  # creating a blank to draw lines on

    lower_color_wall = np.array([100,100,100])
    upper_color_wall = np.array([136,143,146])
    mask_wall = cv.inRange(img, lower_color_wall, upper_color_wall)
    lines_wall = cv.HoughLinesP(mask_wall, 1, np.pi / 180, 3, np.array([]), 100, 50)

    cv.imshow('mask_wall',cv.resize(mask_wall, dsize=None, fx=0.7, fy=0.7)) 

    lower_color_wall = np.array([210,210,210])
    upper_color_wall = np.array([255,255,255])
    mask_write = cv.inRange(img, lower_color_wall, upper_color_wall)
    lines_write = cv.HoughLinesP(mask_write, 1, np.pi / 180, 3, np.array([]), 30, 30)

    cv.imshow('mask_write',cv.resize(mask_write, dsize=None, fx=0.7, fy=0.7)) 

    if lines_wall is not None and lines_write is not None:
        lines = np.append( lines_wall, lines_black, axis=0 )
    elif lines_wall is not None:
        lines = lines_wall
    else:
        lines = lines_write

    lower_color_black = np.array([0,0,0])
    upper_color_black = np.array([2,2,2])
    mask_black = cv.inRange(img, lower_color_black, upper_color_black)
    lines_black = cv.HoughLinesP(mask_black, 1, np.pi / 180, 7, np.array([]), 40, 27)

    cv.imshow('mask_black',cv.resize(mask_black, dsize=None, fx=0.7, fy=0.7)) 

    if lines.any():
        lines = np.append( lines, lines_black, axis=0 )
    else:
        lines = lines_black

    for line in lines:
        x1,y1,x2,y2 = line.ravel()
        cv.line(wall_image,(x1,y1),(x2,y2),(255,255,255),10)

    return lines, wall_image

def getCorners(img):
    corners_image = np.copy(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, 100, 0.001, 3)
    corners = np.int0(corners)

    for corner in corners:
        x,y = corner.ravel()
        cv.circle(corners_image, (x,y), 5, (255,0,0), -1)

    return corners, corners_image



IMAGE_NAME = 'image2.jpeg'
img = cv.imread(IMAGE_NAME, cv.IMREAD_COLOR)

lines, lines_image = getImageWalls(img)
corners, corners_image = getCorners(lines_image)

cv.imshow('img',cv.resize(img, dsize=None, fx=0.7, fy=0.7)) 
cv.imshow('teste',cv.resize(lines_image, dsize=None, fx=0.7, fy=0.7))
cv.imshow('corners',cv.resize(corners_image, dsize=None, fx=0.7, fy=0.7)) 

cv.waitKey(0)
cv.destroyAllWindows()
