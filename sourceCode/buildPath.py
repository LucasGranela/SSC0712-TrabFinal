#Import packages
import cv2 as cv
from nbformat import convert
import numpy as np
from astar import astar

def convertToSceneCoord( imageX, imageY, img_height, img_width ):
    """
    Convert image coordinates to CoppeliaSim coordinates.
    Parameters:
    - imageX: x coordinate on the image (increases from left to right) 
    - imageY: y coordinate on the image (increases from top to bottom)
    - img_height: image height ( img.shape[0] )
    - img_width: image width ( img.shape[1] )
    """

    w = 28.0 # floor size + borders from print
    h = 28.0

    sceneX = -( (imageY*h/img_height) - h/2 )
    sceneY = -( (imageX*w/img_width) - w/2 )

    return np.array( [sceneX, sceneY] )

def convertFromSceneCoord( sceneX, sceneY, img_height, img_width ):
    """
    Convert CoppeliaSim coordinates to image coordinates.
    Parameters:
    - sceneX: x coordinate on the scene
    - sceneY: y coordinate on the scene
    - img_height: image height ( img.shape[0] )
    - img_width: image width ( img.shape[1] )
    """

    w = 28.0 # floor size + borders from print
    h = 28.0

    imageY = int( ((-sceneX)+h/2)*img_height/h )
    imageX = int( ((-sceneY)+w/2)*img_width/w )

    return np.array( [imageX, imageY] )

def getImageWalls(img):
    """
    Generate occupancy grid from scene walls.
    """    

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

    return wall_image

def buildPath( sceneID, originScene, destinationScene ):
    """
    Find path to destination based on scene image.
    Parameters:
    - sceneID: scene number, used to retreive image
    - originScene: pioneer coordinates
    - destinationScene: flag coordinates
    """

    IMAGE_NAME = './image{}.jpeg'.format( sceneID )
    img = cv.imread(IMAGE_NAME, cv.IMREAD_COLOR)

    lines_image = getImageWalls(img)

    # pioneer position
    origin = convertFromSceneCoord( originScene[0], originScene[1], img.shape[0], img.shape[1])
    destination = convertFromSceneCoord( destinationScene[0], destinationScene[1], img.shape[0], img.shape[1])

    cv.circle(img, (origin[0], origin[1]), 5, (0,0,255), -1)
    cv.circle(img, (destination[0], destination[1]), 5, (0,0,255), -1)

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

    path = astar( reduced_grid, (originGrid[1], originGrid[0]), (destinationGrid[1], destinationGrid[0]) )
    path.reverse()

    image_path = []
    for point in path:
        x = int(point[1]*img_width/target)
        y = int(point[0]*img_height/target)
        image_path.append( (x,y) )
        cv.circle(img, (x, y), 5, (0,0,255), -1)

    scene_path = [ convertToSceneCoord( point[0], point[1], img_height, img_width ) for point in image_path ]

    for point in path:
        reduced_image[point[0], point[1], 1] = 255

    # cv.imshow('Original image',cv.resize(img, dsize=None, fx=0.7, fy=0.7)) 
    # cv.imshow('Extracted walls',cv.resize(lines_image, dsize=None, fx=0.7, fy=0.7))
    # cv.imshow('Grid', cv.resize(reduced_image, dsize=(600,600), interpolation=cv.INTER_NEAREST) ) 

    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return scene_path
