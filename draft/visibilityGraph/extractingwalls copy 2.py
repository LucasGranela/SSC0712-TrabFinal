
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

def getImageWalls(img):

    thickness = 10

    wall_image = np.copy(img) * 0  # creating a blank to draw lines on
    rect_image = np.copy(img) * 0
    
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
        cv.line(wall_image,(x1,y1),(x2,y2),(255,255,255),thickness)

    cv.imshow('lines',wall_image) 

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

        rect = [p1,p2,p3,p4]
        rects.append(rect)

        cv.rectangle(rect_image, (p1[0,0], p1[0,1]), (p4[0,0], p4[0,1]), (255,255,255))

    rects = np.array(rects).reshape(-1, 4, 2)

    # cv.imshow('polygons',poly_image) 
    # cv.imshow('contour',contour_image) 

    return np.copy(wall_image), np.copy(rect_image), rects

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

line_image, rect_image, rects = getImageWalls(img)

# Create a list of polygons, where each polygon corresponds to a shape
polygons = []
for rect in rects[0:10]:
    # print("Rectangle:",rect)
    polygon = []
    for point in rect:
        # print("Point:", point)
        polygon.append( vg.Point(point[0], point[1]) )
    polygons.append(polygon)

print(polygons)

# Start building the visibility graph 
g = vg.VisGraph()
# g.build(polygons)
g.build( polygons[0:1] )
g.save('./visibility_graph.pk1')

shortest = g.shortest_path( vg.Point(1.5,0.0), vg.Point(4.0, 6.0) )
print( shortest )

cv.circle(img, (0,0), 5, (255,0,0), -1)
cv.circle(img, (img.shape[1],0), 5, (0,255,0), -1)

cv.circle(img, convertFromSceneCoord(-9.81, -0.40, img.shape[0], img.shape[1]), 10, (0,0,255), -1)

cv.imshow('original image',img) 
cv.imshow('rectangles', rect_image)
# cv.imshow('corners',final_image) 

print("Altura (height): %d pixels" % (img.shape[0]))
print("Largura (width): %d pixels" % (img.shape[1]))

cv.waitKey(0)

# Clean up
cv.destroyAllWindows()