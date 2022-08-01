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

    lower_color_wall = np.array([210,210,210])
    upper_color_wall = np.array([255,255,255])
    mask_write = cv.inRange(img, lower_color_wall, upper_color_wall)
    lines_write = cv.HoughLinesP(mask_write, 1, np.pi / 180, 3, np.array([]), 30, 30)

    if lines_wall is not None and lines_write is not None:
        lines = np.append( lines_wall, lines_write, axis=0 )
    elif lines_wall is not None:
        lines = lines_wall
    else:
        lines = lines_write

    lower_color_black = np.array([0,0,0])
    upper_color_black = np.array([2,2,2])
    mask_black = cv.inRange(img, lower_color_black, upper_color_black)
    lines_black = cv.HoughLinesP(mask_black, 1, np.pi / 180, 7, np.array([]), 40, 27)

    if lines.any() and lines_black is not None:
        lines = np.append( lines, lines_black, axis=0 )
    elif lines_black is not None:
        lines = lines_black

    for line in lines:
        x1,y1,x2,y2 = line.ravel()
        cv.line(wall_image,(x1,y1),(x2,y2),(255,255,255),10)

    return wall_image

def buildPath( sceneID, originScene, destinationScene, showPath=False ):
    """
    Find path to destination based on scene image.
    Parameters:
    - sceneID: scene number, used to retreive image
    - originScene: pioneer coordinates
    - destinationScene: flag coordinates
    - showPath: whether to display images
    """

    IMAGE_NAME = './image{}.jpeg'.format( sceneID )
    img = cv.imread(IMAGE_NAME, cv.IMREAD_COLOR)

    # extract walls (ocuppancy grid)
    lines_image = getImageWalls(img)

    # get pioneer and flag position
    origin = convertFromSceneCoord( originScene[0], originScene[1], img.shape[0], img.shape[1])
    destination = convertFromSceneCoord( destinationScene[0], destinationScene[1], img.shape[0], img.shape[1])

    # plot origin and destination on image
    cv.circle(img, (origin[0], origin[1]), 5, (0,0,255), -1)
    cv.circle(img, (destination[0], destination[1]), 5, (0,0,255), -1)

    # reduce occupancy grid dimensionality for A*
    target=50
    reduced_grid = cv.resize(lines_image, dsize=(target,target), interpolation=cv.INTER_AREA)
    reduced_grid = np.sum(reduced_grid, axis=-1)
    reduced_grid = np.where(reduced_grid > 50, 1, 0)

    # generate image of reduced grid (blue = walls)
    reduced_image = np.zeros( shape=(50,50,3), dtype=np.uint8 )
    reduced_image[:, :, 0] = 255*reduced_grid

    # determine origin and destination position on the grid
    img_height = img.shape[0]
    img_width  = img.shape[1]
    originGrid = np.array( [int(origin[0]*target/img_width), int(origin[1]*target/img_height)] )
    destinationGrid = np.array( [int(destination[0]*target/img_width), int(destination[1]*target/img_height)] )

    # plot origin and destination on the grid (red)
    reduced_image[originGrid[1], originGrid[0], 2] = 255
    reduced_image[destinationGrid[1], destinationGrid[0], 2] = 255

    # find path using A* algorithm
    path = astar( reduced_grid, (originGrid[1], originGrid[0]), (destinationGrid[1], destinationGrid[0]) )
    path.reverse()

    # plot path on the original image
    image_path = []
    for point in path:
        # convert to original image coordinates
        x = int(point[1]*img_width/target)
        y = int(point[0]*img_height/target)
        image_path.append( (x,y) )
        cv.circle(img, (x, y), 5, (0,0,255), -1)

    # convert to scene coordinates
    scene_path = [ convertToSceneCoord( point[0], point[1], img_height, img_width ) for point in image_path ]

    # plot path on the grid (green)
    for point in path:
        reduced_image[point[0], point[1], 1] = 255

    if showPath:
        cv.imshow('Extracted walls (press any key to continue)',cv.resize(lines_image, dsize=None, fx=0.7, fy=0.7))
        cv.imshow('A* (press any key to continue)', cv.resize(reduced_image, dsize=(600,600), interpolation=cv.INTER_NEAREST) ) 
        cv.imshow('Waypoint (press any key to continue)',cv.resize(img, dsize=None, fx=0.7, fy=0.7)) 

        cv.waitKey(0)
        cv.destroyAllWindows()

    return scene_path
