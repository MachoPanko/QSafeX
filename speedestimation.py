from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2

def speedestimation(source):
    ## this part is to set up the yolo library with official weights, after that can run normally using custom weights
    ## run this on if u are encounterring the following : WARNING ⚠️ weights/barrel.pt appears to require 'ultralytics.yolo', which is not in ultralytics requirements.
    # model = YOLO("yolov8n.pt")
    # vcap = cv2.VideoCapture(source)
    # ret, image = vcap.read()
    # model(image)
    # vcap.release()
    # cv2.destroyAllWindows()
    model = YOLO("weights/best.pt")
    names = model.model.names
    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), "Error reading video file"
    # Video writer


    line_pts = [(0, 360), (1280, 360)]

    # Init speed-estimation obj
    speed_obj = speed_estimation.SpeedEstimator()
    speed_obj.set_args(reg_pts=line_pts,
                       names=names,
                       view_img=True)

    while cap.isOpened():

        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        tracks = model.track(im0, persist=True, show=False)

        im0 = speed_obj.estimate_speed(im0, tracks)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
