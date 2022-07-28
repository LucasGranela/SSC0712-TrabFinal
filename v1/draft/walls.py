from locale import DAY_1
import numpy as np
import cv2 as cv

def convertToSceneCoord( imageX, imageY, img_height, img_width ):
    """
    Convert image coordinates to CoppeliaSim coordinates.
    Parameters:
    - imageX: x coordinate on the image (increases from left to right) 
    - imageY: y coordinate on the image (increases from top to bottom)
    - img_height: image height ( img.shape[0] )
    - img_width: image width ( img.shape[1] )
    """

    w = 25.0 # CoppeliaSim scene width
    h = 25.0 # CoppeliaSim scene height

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

    w = 25.0 # CoppeliaSim scene width
    h = 25.0 # CoppeliaSim scene height

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
    # print( lines.shape )

    # saved_lines = np.copy(lines)

    # new_lines = np.empty( (1,1,4), dtype=int )

    # thresh = 10

    # for i in range(lines.shape[0]):
    #     if sum(lines[i]) == 0:
    #         continue

    #     for j in range(i+1, lines.shape[0]):
    #         if sum( lines[j] ) == 0:
    #             continue

    #         dyi = lines[i, 0, 1] - lines[i, 0, 3] 
    #         dxi = lines[i, 0, 0] - lines[i, 0, 2] 
    #         dyj = lines[j, 0, 1] - lines[j, 0, 3]
    #         dxj = lines[j, 0, 0] - lines[j, 0, 2] 

    #         slopei = np.arctan2( dyi, dxi )
    #         slopej = np.arctan2( dyj, dxj )

    #         if np.abs(slopei - slopej) < 0.2:
    #             dxij1 = np.abs( min( lines[i, 0, 0], lines[i, 0, 2]  ) - max( lines[j, 0, 0], lines[j, 0, 2] ) )
    #             dxij2 = np.abs( max( lines[i, 0, 0], lines[i, 0, 2]  ) - min( lines[j, 0, 0], lines[j, 0, 2] ) )
    #             dxij = min( dxij1, dxij2)

    #             dyij1 = np.abs( min( lines[i, 0, 1], lines[i, 0, 3]  ) - max( lines[j, 0, 1], lines[j, 0, 3] ) )
    #             dyij2 = np.abs( max( lines[i, 0, 1], lines[i, 0, 3]  ) - min( lines[j, 0, 1], lines[j, 0, 3] ) )
    #             dyij = min( dyij1, dyij2)

    #             if dxij<thresh and dyij<thresh:
    #                 # merge lines
    #                 pi1 = lines[i, 0, [0,1]]
    #                 pi2 = lines[i, 0, [2,3]]
    #                 pj1 = lines[j, 0, [0,1]]
    #                 pj2 = lines[j, 0, [2,3]]
    #                 x1 = 
    #                 y1
    #                 x2
    #                 y2
    #                 new_line = np.array( [[]] )
    
    for line in lines:
        x1,y1,x2,y2 = line.ravel()
        cv.line(wall_image,(x1,y1),(x2,y2),(255,255,255),10)

    return lines, wall_image


if __name__ == '__main__':
    
    IMAGE_NAME = './image.jpeg'
    img = cv.imread(IMAGE_NAME, cv.IMREAD_COLOR)

    lines, lines_image = getImageWalls(img)

    cv.imshow('Original image', cv.resize(img, dsize=None, fx=0.7, fy=0.7)) 
    cv.imshow('Lines', cv.resize(lines_image, dsize=None, fx=0.7, fy=0.7))

    cv.waitKey(0)
    cv.destroyAllWindows()