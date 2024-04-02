import cv2
import torch


def define_rect(image):
    """
    Define a rectangular window by click and drag your mouse.

    Parameters
    ----------
    image: Input image.
    """

    clone = image.copy()
    rect_pts = [(0,0),(0,0)]  # Starting and ending points
    win_name = "image"  # Window name

    def select_points(event, x, y, flags, param):

        nonlocal rect_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(x, y)]

        if event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((x, y))

            # draw a rectangle around the region of interest
            cv2.rectangle(clone, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points)

    while True:
        # display the image and wait for a keypress
        cv2.imshow(win_name, clone)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"):  # Hit 'r' to replot the image
            clone = image.copy()

        elif key == ord("c"):  # Hit 'c' to confirm the selection
            break

    # close the open windows
    cv2.destroyWindow(win_name)

    return rect_pts


##List of Lists, each element is a list containing 2 tuples of x,y coord for top left and btm right corners.
def define_rect_all(image, num_functions=3):
    regions = []
    for i in range(num_functions):
        regions.append(define_rect(image))
    return regions

def boxes_in_region(boxes, region_of_interest):
    important_boxes = torch.tensor([])
    important_cls = torch.tensor([])
    cls = boxes.cls
    xyxy = boxes.xyxy
    for i in range(len(xyxy)):
        if region_of_interest[0][0] < (xyxy[i][0]+xyxy[i][2])//2 < region_of_interest[1][0] and region_of_interest[0][1] < (xyxy[i][1]+xyxy[i][3])//2 < region_of_interest[1][1]:
            # print(cls[i].dtype)
            # print(cls[i].shape)
            # print(xyxy[i].dtype)
            # print(xyxy[i].shape)
            important_boxes = torch.cat((important_boxes, xyxy[i].unsqueeze(0)), 0)
            important_cls = torch.cat((important_cls, cls[i].unsqueeze(0)), 0)
    # print(important_boxes)
    # print(important_cls)
    return important_boxes, important_cls
