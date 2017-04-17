import cv2
import numpy as np
import sys
import pydarknet

pydarknet.load("cfg/tiny-yolo-voc.cfg", "tiny-yolo-voc.weights")

def detect_frame(img):
    (h, w, c) = img.shape
    ret = pydarknet.detect(img.data, w, h, c, 0.1)
    return ret

def draw_dets(img, dets):
    for (name, thres, left, right, top, bot) in dets:
        p1 = (left, top)
        p2 = (right, bot)
        cv2.putText(img, name + ' : %.2f' % thres,(left+5, top + 20),0,0.5,(0,0,0), 2)
        cv2.rectangle(img, p1, p2, (255, 0, 0), 2)
    return img

def main():
    video = sys.argv[1] 
    rotation = 0
    if len(sys.argv) > 2:
        rotation = int(sys.argv[2])

    cap = cv2.VideoCapture(video)
    if not cap.isOpened:
        print('Cannot open video file', video)
        return

    tots = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    fps = round(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    
    ret, frame = cap.read()

    rows = frame.shape[0]
    cols = frame.shape[1]
    print('video info: fps:%d %dx%d' %(fps, cols, rows))

    # save video
    #fourcc = cv2.cv.CV_FOURCC(*'X264')
    fourcc = cv2.cv.CV_FOURCC(*'XVID')

    resample = max(1, round(max(cols, rows)/1024))
    frame_size = (cols, rows)
    reframe_size = (int(cols/resample), int(rows/resample))
    out_vsize = reframe_size 
    if rotation == 90 or rotation == 270:
        out_vsize = (reframe_size[1], reframe_size[0])

    outVideo = cv2.VideoWriter('output.mp4',fourcc, fps, out_vsize)

    dets = None
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
    i = 0
    while True:
        ret, frame = cap.read()
        if frame == None:
            break
        frame = cv2.resize(frame, reframe_size, interpolation = cv2.INTER_AREA)
        if rotation == 90 or rotation == 270:
            H = cv2.getRotationMatrix2D((reframe_size[0]/2, reframe_size[1]/2), rotation,1)
            frame = cv2.warpAffine(frame,H, out_vsize)

        if i% fps == 0:
            dets = detect_frame(frame)

        frame = draw_dets(frame, dets)
        i += 1
        #cv2.imwrite('frame.jpg', frame)
        outVideo.write(frame)
    outVideo.release()


if __name__ == '__main__':
    main()
