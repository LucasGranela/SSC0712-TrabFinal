import cv2 as cv

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

IMAGE_NAME = './image.jpeg'
img = cv.imread(IMAGE_NAME, cv.IMREAD_COLOR)

print("Image shape: ", img.shape)

# BGR
cv.circle(img, (0,0), 5, (255,0,0), -1)
cv.circle(img, (img.shape[1],0), 5, (0,255,0), -1)
cv.circle(img, (0,img.shape[0]), 5, (0,0,255), -1)

cv.circle(img, convertFromSceneCoord(-9.81, -0.4, img.shape[0], img.shape[1]), 5, (0,0,255), -1)

cv.imshow( "Original image", cv.resize(img, dsize=None, fx=0.7, fy=0.7) ) 

cv.waitKey(0)
cv.destroyAllWindows()

# BGR