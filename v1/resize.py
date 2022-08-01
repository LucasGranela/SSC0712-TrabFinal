import cv2 as cv

IMAGE_NAME = 'image3.jpeg'

img = cv.imread( IMAGE_NAME, cv.IMREAD_COLOR )
cv.imshow('Image',cv.resize(img, dsize=None, fx=0.7, fy=0.7)) 

k = -1
border=1
imgx = img.shape[0]
imgy = img.shape[1]
while k != 13:
    k = cv.waitKey( 0 )

    if k == 82: # upkey
        border += border
    elif k == 84: # downkey
        border -= (border > 0)

    print(border)
    resized_img = img[border:imgx-border, border:imgy-border, :]
    cv.imshow('Image',cv.resize(resized_img, dsize=None, fx=0.7, fy=0.7)) 


cv.imwrite( '{}_resized.jpeg'.format(IMAGE_NAME), resized_img )
cv.destroyAllWindows()