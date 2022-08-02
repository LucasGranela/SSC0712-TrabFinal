
#Import packages
import cv2 as cv
import numpy as np
import pyvisgraph as vg

def convertToSceneCoord( x, y, img_height, img_width ):
    None

def convertFromSceneCoord( x, y ):
    None

def getImageWalls(img):

    thickness = 10

    wall_image = np.copy(img) * 0  # creating a blank to draw lines on
    poly_image = np.copy(wall_image)
    contour_image = cv.cvtColor(wall_image, cv.COLOR_BGR2GRAY)
    # print( wall_image.shape )

    lower_color_wall = np.array([100,100,100])
    upper_color_wall = np.array([136,143,146])
    mask_wall = cv.inRange(img, lower_color_wall, upper_color_wall) 

    # cv.imshow('mask 1',mask_wall) 

    lines_wall = cv.HoughLinesP(mask_wall, 1, np.pi / 180, 3, np.array([]), 100, 50)

    # print( len(lines_wall) )

    # for line in lines_wall:
    #     x1,y1,x2,y2 = line.ravel()
    #     cv.line(wall_image,(x1,y1),(x2,y2),(255,255,255),thickness)

    lower_color_black = np.array([0,0,0])
    upper_color_black = np.array([2,2,2])
    mask_black = cv.inRange(img, lower_color_black, upper_color_black)

    # cv.imshow('mask 2',mask_black) 

    lines_black = cv.HoughLinesP(mask_black, 1, np.pi / 180, 7, np.array([]), 30, 40)

    # print( len(lines_black) )
    # print( type(lines_black) )
    lines = np.append( lines_wall, lines_black, axis=0 )


    scale=5
    polygons=[]
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

        polygon = [p1,p2,p3,p4]
        polygons.append(polygon)

        cv.rectangle(poly_image, (p1[0,0], p1[0,1]), (p4[0,0], p4[0,1]), (255,255,255))

    for line in lines:
        x1,y1,x2,y2 = line.ravel()
        cv.line(wall_image,(x1,y1),(x2,y2),(255,255,255),thickness)

    gray_wall_image = cv.cvtColor(wall_image, cv.COLOR_BGR2GRAY)
    contours,_ = cv.findContours(gray_wall_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        contour = contour.reshape((-1,1,2))
        cv.polylines(contour_image,[contour],True,(255,255,255))

    # cv.imshow('polygons',poly_image) 
    # cv.imshow('contour',contour_image) 

    return np.copy(wall_image), np.copy(poly_image), np.array(polygons)

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

line_image, rect_image, rectangles = getImageWalls(img)
# final_image, corners = getCorners(line_image)

rectangles = rectangles.reshape( -1, 4, 2 )
print( rectangles.shape )
print( rectangles[0] )

# Create a list of polygons, where each polygon corresponds to a shape
polygons = []
for rect in rectangles:
    polygon = []
    for point in rect:
        polygon.append(vg.Point(point[0], point[1]))
    polygons.append(polygon)

# Start building the visibility graph 
graph = vg.VisGraph()
graph.build(polygons)
graph.save('./visibility_graph.pk1')

shortest = graph.shortest_path( vg.Point(1.5,0.0), vg.Point(4.0, 6.0) )
print( shortest )

cv.circle(img, (0,0), 5, (255,0,0), -1)
cv.circle(img, (img.shape[0],0), 5, (0,255,0), -1)

cv.imshow('original image',img) 
cv.imshow('rectangles', rect_image)
# cv.imshow('corners',final_image) 

print("Altura (height): %d pixels" % (img.shape[0]))
print("Largura (width): %d pixels" % (img.shape[1]))

cv.waitKey(0)

# Clean up
cv.destroyAllWindows()