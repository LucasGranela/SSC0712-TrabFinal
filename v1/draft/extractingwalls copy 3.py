
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

        cv.rectangle(rect_image, (p1[0,0], p1[0,1]), (p4[0,0], p4[0,1]), (255,255,255), -1)

    rects = np.array(rects).reshape(-1, 4, 2)

    return np.copy(rect_image), rects

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

rect_image, rects = getImageWalls(img)

cv.imshow('Original image', cv.resize(img, dsize=None, fx=0.7, fy=0.7)) 
cv.imshow('Extracted rectangles', cv.resize(rect_image, dsize=None, fx=0.7, fy=0.7))


gray_rect_image = cv.cvtColor(rect_image, cv.COLOR_BGR2GRAY)
contours,_ = cv.findContours(gray_rect_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print( len(contours) )

contour_image = np.copy(gray_rect_image) * 0
for contour in contours[0:1]:
    contour = contour.reshape((-1,1,2))
    cv.polylines(contour_image,[contour],True,(255,255,255))

cv.imshow('Contour', cv.resize(contour_image, dsize=None, fx=0.7, fy=0.7))

# # Create a list of polygons, where each polygon corresponds to a shape
# polygons = []
# for rect in rects[0:10]:
#     # print("Rectangle:",rect)
#     polygon = []
#     for point in rect:
#         # print("Point:", point)
#         polygon.append( vg.Point(point[0], point[1]) )
#     polygons.append(polygon)

# print(polygons)

# # Start building the visibility graph 
# g = vg.VisGraph()
# # g.build(polygons)
# g.build( polygons[0:1] )
# g.save('./visibility_graph.pk1')

# shortest = g.shortest_path( vg.Point(1.5,0.0), vg.Point(4.0, 6.0) )
# print( shortest )


cv.waitKey(0)

# Clean up
cv.destroyAllWindows()