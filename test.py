import pydarknet
import cv2
import sys
import os

def detect_video(video_file, threshold = 0.5, output_file ='./output.avi', detect_rate=0):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened:
        print('Cannot open video file', video)
        return

    # video info
    tots = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    fps = round(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    
    ret, frame = cap.read()
    rows = frame.shape[0]
    cols = frame.shape[1]
    print('video info: fps:%d %dx%d' %(fps, cols, rows))
    if detect_rate <= 0:
        detect_rate = 1;

    cv2.namedWindow('video', cv2.cv.CV_WINDOW_NORMAL)
    cv2.moveWindow('video', 200, 150)

    # save video
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    frame_size = (cols, rows)
    outVideo = cv2.VideoWriter(output_file,fourcc, fps, frame_size)

    dets = None
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
    i = 0
    while True:
        ret, frame = cap.read()
        if frame == None:
            break
        if i % detect_rate == 0:
            dets = pydarknet.detect_image(frame)
        frame = pydarknet.draw_dets(frame, dets, text=str(i))
        cv2.imshow("video", frame)
        cv2.waitKey(1)
        outVideo.write(frame)
        i += 1
    outVideo.release()

if __name__ == '__main__':
    #pydarknet.load("cfg/tiny-yolo-voc.cfg", "tiny-yolo-voc.weights")
    pydarknet.load("cfg/yolo.cfg", "yolo.weights")

    input_path = 'data/dog.jpg'
    if len(sys.argv) > 1:
        input_path = sys.argv[1]

    ext = os.path.splitext(input_path)[1]
    ext = ext.lower()

    if ext == '.png' or ext == '.jpg' or ext == '.jpeg':
        #ret = pydarknet.detect_file(input_path)
        img = cv2.imread(input_path)
        ret = pydarknet.detect_image(img)

        img = pydarknet.draw_dets(img, ret, text='100')

        outfile = os.path.splitext(input_path)[0] + '_detect' + ext
        cv2.imwrite(outfile, img)
        cv2.imshow("", img)
        cv2.waitKey()
    else:
        detect_video(input_path, detect_rate=10)

