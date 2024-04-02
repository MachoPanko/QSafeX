from ultralytics import YOLO
import ultralytics
import cv2

import calibrateregions


def counting(source, region_of_interest):
    # this part is to set up the yolo library with official weights, after that can run normally using custom weights
    # run this on if u are encounterring the following : WARNING ⚠️ weights/barrel.pt appears to require 'ultralytics.yolo', which is not in ultralytics requirements.
    model = YOLO("yolov8n.pt")
    model("input_media/black_screen.png")
    #
    #

    model = YOLO("weights/barrel.pt")
    number_of_barrels = 0
    avg_count = 0
    n_iterations = 0
    results = model(source=source, show=True, stream=True)
    for result in results:

        boxes = result.boxes
        print(boxes)
        xyxy, classes = calibrateregions.boxes_in_region(boxes, region_of_interest)
        print( xyxy)
        number_of_barrels = len(classes)
        if (number_of_barrels != 0):
            n_iterations += 1
            avg_count = (avg_count * (n_iterations - 1) + number_of_barrels) / n_iterations
        print("number of barrels in this image:", number_of_barrels)
        print("average number of barrels:", avg_count)
        number_of_barrels = 0

    return avg_count
    #### This whole part is for counting total number of instances seen (throughout the video)####rtsp://admin:admin@192.168.68.69:6968/h264.sdp
    # model = YOLO("weights/barrel.pt")
    # cap = cv2.VideoCapture(source)
    # assert cap.isOpened(), "Error reading video file"
    # w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    # # Define region points
    # region_points = [(0, 0), (w, 0), (w, h), (0, h)]
    #
    # # Video writer
    # video_writer = cv2.VideoWriter("output_media/object_counting_output.avi",
    #                        cv2.VideoWriter_fourcc(*'mp4v'),
    #                        fps,
    #                        (w, h))
    #
    # # Init Object Counter
    # counter = object_counter.ObjectCounter()
    # counter.set_args(view_img=True,
    #                  reg_pts=region_points,
    #                  classes_names=model.names,
    #                  draw_tracks=True)
    #
    # while cap.isOpened():
    #     success, im0 = cap.read()
    #     if not success:
    #         print("Video frame is empty or video processing has been successfully completed.")
    #         break
    #     tracks = model.track(im0, persist=True, show=False)
    #
    #     im0 = counter.start_counting(im0, tracks)
    #     video_writer.write(im0)
    #
    # cap.release()
    # video_writer.release()
    # cv2.destroyAllWindows()
