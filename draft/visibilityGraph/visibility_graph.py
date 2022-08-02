# só a ideia, tem que arrumar a recursão

#Import packages
import cv2 as cv
from nbformat import convert
import numpy as np
from astar import astar

def convertToSceneCoord( imageX, imageY, img_height, img_width ):
    w = 28.0
    h = 28.0

    sceneX = -( (imageY*h/img_height) - h/2 )
    sceneY = -( (imageX*w/img_width) - w/2 )

    return np.array( [sceneX, sceneY] )

def convertFromSceneCoord( sceneX, sceneY, img_height, img_width ):
    w = 28.0
    h = 28.0

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

    return lines, wall_image

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

def reduce(img):
    target = 50

###

IMAGE_NAME = './image.jpeg'
img = cv.imread(IMAGE_NAME, cv.IMREAD_COLOR)

lines, lines_image = getImageWalls(img)
corners, corners_image = getCorners(lines_image)

origin = convertFromSceneCoord( -9.8195, -0.40, img.shape[0], img.shape[1])
destination = convertFromSceneCoord( +11.625, -11.65, img.shape[0], img.shape[1])

cv.circle(img, (origin[0], origin[1]), 5, (0,0,255), -1)
cv.circle(img, (destination[0], destination[1]), 5, (0,0,255), -1)

# lines = lines.reshape( -1, 4 )
# corners = corners.reshape( -1, 2 )

# save_corners = np.copy( corners )

# corners = np.append( corners, np.array( [destination] ), axis=0 )

# path = [origin]

# findPath( path.copy(), destination, np.copy(corners), lines )

target=50
reduced_grid = cv.resize(lines_image, dsize=(target,target), interpolation=cv.INTER_AREA)
reduced_grid = np.sum(reduced_grid, axis=-1)
reduced_grid = np.where(reduced_grid > 50, 1, 0)
reduced_image = np.zeros( shape=(50,50,3), dtype=np.uint8 )
reduced_image[:, :, 0] = 255*reduced_grid

img_height = img.shape[0]
img_width  = img.shape[1]

originGrid = np.array( [int(origin[0]*target/img_width), int(origin[1]*target/img_height)] )
destinationGrid = np.array( [int(destination[0]*target/img_width), int(destination[1]*target/img_height)] )

reduced_image[originGrid[1], originGrid[0], 2] = 255
reduced_image[destinationGrid[1], destinationGrid[0], 2] = 255

# 0 - empty
# 1 - obstacle
# 2 - starting point
# 3 - goal
# reduced_grid[originGrid[1], originGrid[0]] = 2
# reduced_grid[destinationGrid[1], destinationGrid[0]] = 3

path = astar( reduced_grid, (originGrid[1], originGrid[0]), (destinationGrid[1], destinationGrid[0]) )

for point in path:
    reduced_image[point[0], point[1], 1] = 255

cv.imshow('Original image',cv.resize(img, dsize=None, fx=0.7, fy=0.7)) 
cv.imshow('Extracted walls',cv.resize(lines_image, dsize=None, fx=0.7, fy=0.7))
cv.imshow('Grid', cv.resize(reduced_image, dsize=(600,600), interpolation=cv.INTER_NEAREST) )
cv.imshow('Corners',cv.resize(corners_image, dsize=None, fx=0.7, fy=0.7)) 

cv.waitKey(0)
cv.destroyAllWindows()
