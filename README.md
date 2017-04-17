![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

#Darknet-python#

add python api for: https://github.com/pjreddie/darknet

deps: opencv2 python2

````
import pydarknet
import cv2

pydarknet.load("cfg/tiny-yolo-voc.cfg", "tiny-yolo-voc.weights")
#pydarknet.load("cfg/yolo.cfg", "yolo.weights")

#pydarknet.detect_file('data/dog.jpg')
img = cv2.imread('data/dog.jpg')

(h, w, c) = img.shape
ret = pydarknet.detect(img.data, w, h, c, 0.1)
for (name, thres, left, right, top, bot) in ret:
    p1 = (left, top)
    p2 = (right, bot)
    thick = max(1, int((h + w) // 300))
    # cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, name + ' : %.2f' % thres,(left+5, top - 7),0,0.5,(0,0,0), 2)
    cv2.rectangle(img, p1, p2, (255, 0, 0), 2)

cv2.imwrite("tmp.jpg", img)
cv2.imshow("", img)
````
