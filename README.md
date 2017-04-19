![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

#Darknet Python#

python detect wraper Darknet: https://github.com/pjreddie/darknet

requirements:

 - opencv2
 - python2

````
import pydarknet
import cv2

#pydarknet.load("cfg/tiny-yolo-voc.cfg", "tiny-yolo-voc.weights")
pydarknet.load("cfg/yolo.cfg", "yolo.weights")

#pydarknet.detect_file('data/dog.jpg')

img = cv2.imread('data/dog.jpg')
dets= pydarknet.detect_image(img)
img = pydarknet.draw_dets(img, dets, text='dog')

cv2.imwrite("detect.jpg", img)
cv2.imshow("", img)
````
