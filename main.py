import cv2

import calibrateregions
import unauthorized_access
import ultralytics
from ultralytics import YOLO
import userSelectionHandler
import telegram
import asyncio
import signalTelegram
from IPython import display
from ultralytics import YOLO
from PIL import Image
import ultralytics
from roboflow import Roboflow
import matplotlib.pyplot as plt
import numpy as np

def main():

    model = YOLO("yolov8s.pt")
    model("input_media/black_screen.png")
    model = YOLO("weights/Goodweights/humans/humanv11.pt")
    human_results = model(source="input_media/brani_ppe_test.jpg")
    for human_result in human_results:
        # model = YOLO("weights/barrel_v2.pt")
        model = YOLO("weights/snehilsanyal-constructionn-site-safety-ppe.pt")
        orig_img = human_result.orig_img
        for x1, y1, x2, y2 in human_result.boxes.xyxy:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_img = orig_img[y1:y2, x1:x2]
            # model = YOLO("weights/snehilsanyal-constructionn-site-safety-ppe.pt")
            ppe_results = model(cropped_img, imgsz=160, classes=[0,5, 7], show=True)
            print(model.model_name)

    # model = YOLO('weights/humanv12.pt')
    # results = model("input_media/brani_ppe_test.jpg")
    # model = YOLO('weights/snehilsanyal-constructionn-site-safety-ppe.pt')
    # results1 = model(results[0].orig_img, save = True)



    #
    # userSelectionHandler.userSelectionHandler()

    #signalTelegram.sendMessage()
    # switcher ={
    #     1: detectionModes.UnauthorizedAccess(source=rtsp_url, model=model),
    #     2: detectionModes.counting(source=rtsp_url, model=model)
    # }
    # switcher.get(m)
if __name__ == '__main__':

    main()
# Replace the below URL with your RTSP stream URL including the username and password
# rtsp_url = 'rtsp://admin:admin@10.10.9.177:6968/h264.sdp'
# model = YOLO("weights/best(2feb).pt")
# unauthorized_access.UnauthorizedAccess(rtsp_url, model)
# mode = int(input("What detection mode would you like? \n (1) Unauthorized Access \n (2) Speeding"))
# switch(mode)

# detectionModes.UnauthorizedAccess(rtsp_url, model)




#
# detectionModes.speeding(results, line=0, names=model.names)

# # Start capturing video from the URL
# cap = cv2.VideoCapture(rtsp_url)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#     results = model(frame)
#     annotated_frame = results[0].plot()
#     # Display the resulting frame
#     cv2.imshow('Frame', annotated_frame)
#
#     # Press Q on keyboard to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
