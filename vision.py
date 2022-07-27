#Import packages
import cv2 as cv
import numpy as np

def getImageWalls(img):
    wall_image = np.copy(img) * 0  # creating a blank to draw lines on
    lower_color_wall = np.array([100,100,100])
    upper_color_wall = np.array([136,143,146])
    mask_wall = cv.inRange(img, lower_color_wall, upper_color_wall) 

    lines_wall = cv.HoughLinesP(mask_wall, 1, np.pi / 180, 3, np.array([]), 100, 50)

    for line in lines_wall:
        x1,y1,x2,y2 = line.ravel()
        cv.line(wall_image,(x1,y1),(x2,y2),(255,255,255),10)

    lower_color_black = np.array([0,0,0])
    upper_color_black = np.array([2,2,2])
    mask_black = cv.inRange(img, lower_color_black, upper_color_black)

    lines_black = cv.HoughLinesP(mask_black, 1, np.pi / 180, 7, np.array([]), 30, 40)

    for line in lines_black:
        x1,y1,x2,y2 = line.ravel()
        cv.line(wall_image,(x1,y1),(x2,y2),(255,255,255),10)

    return np.copy(wall_image)

def getCorners(img):
    corners_image = np.copy(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, 50, 0.001, 5)
    corners = np.int0(corners)

    print(corners)

    for corner in corners:
        x,y = corner.ravel()
        cv.circle(corners_image, (x,y), 5, (255,0,0), -1)

    return corners_image , corners 



IMAGE_NAME = './image.jpeg'
img = cv.imread(IMAGE_NAME, cv.IMREAD_COLOR)

line_image = getImageWalls(img)
final_image, corners = getCorners(line_image)

cv.imshow('img',img) 
cv.imshow('teste',line_image)
cv.imshow('corners',final_image) 

print("Altura (height): %d pixels" % (img.shape[0]))
print("Largura (width): %d pixels" % (img.shape[1]))

cv.waitKey(0)

# Clean up
cv.destroyAllWindows()
