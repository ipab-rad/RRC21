# some general imports
import numpy as np
import cv2
import pathlib

# trifinger imports
import trifinger_cameras.py_tricamera_types as tricamera
from trifinger_cameras.utils import convert_image
from trifinger_object_tracking.py_lightblue_segmenter import segment_image

class DicePose:
    def __init__(self):
        pass

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    return labeled_img

def main():
    # retreive same timestamped(closest) frames from the 3 cameras

    data_dir = "/home/aditya/real_output/45608/"
    camera_data = "/home/aditya/real_output/45608/camera_data.dat"

    # initialise log reader
    log_reader = tricamera.LogReader(camera_data)
    for observation in log_reader.data:
        image60 = convert_image(observation.cameras[0].image, format="bgr")
        image180 = convert_image(observation.cameras[1].image, format="bgr")
        image300 = convert_image(observation.cameras[2].image, format="bgr")

        # try and overlay the segemented image over the original image
        mask60 = segment_image(image60)
        mask180 = segment_image(image180)
        mask300 = segment_image(image300)


        # overlay mask over image
        # this is not working, needs more attention
        mod60 = cv2.cvtColor(image60, cv2.COLOR_BGR2BGRA)
        mod60[:, :, 3] = mask60.astype(np.uint8)
        mod180 = cv2.cvtColor(image180, cv2.COLOR_BGR2BGRA)
        mod180[:, :, 3] = mask180.astype(np.uint8)
        mod300 = cv2.cvtColor(image300, cv2.COLOR_BGR2BGRA)
        mod300[:, :, 3] = mask300.astype(np.uint8)

        # perform connected components on the binary mask image
        out60 = cv2.connectedComponentsWithStats(mask60, 4, cv2.CV_32S)
        print("components in 60: {}".format(out60[0]))
        out180 = cv2.connectedComponentsWithStats(mask180, 4, cv2.CV_32S)
        print("components in 180: {}".format(out180[0]))
        out300 = cv2.connectedComponentsWithStats(mask300, 4, cv2.CV_32S)
        print("components in 300: {}".format(out300[0]))


        cv2.imshow("camera60", imshow_components(out60[1]))
        cv2.imshow("camera180", imshow_components(out180[1]))
        cv2.imshow("camera300", imshow_components(out300[1]))

        cv2.waitKey(100)
    # open the segmentation maps
    # draw connected components on the segmentation maps
    # visualise

if __name__ == "__main__":
    main()
