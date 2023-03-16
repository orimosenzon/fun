import cv2 
cv2.resize(src, dsize, fx, fy, interpolation)

'''
src	- The file path in which the input image resides.
dsize -	The size of the output image, which adheres to the syntax (width, height).
fx	- The scale factor for the X axis.
fy	- The scale factor for the Y axis.
interpolation - The technique for adding or removing pixels during the resizing process. The default is cv2.INTER_LINEAR.
'''