# some general imports
import numpy as np
import cv2
import pathlib
import sys

# trifinger imports
import trifinger_cameras.py_tricamera_types as tricamera
from trifinger_cameras.utils import convert_image
from trifinger_object_tracking.py_lightblue_segmenter import segment_image

class DicePose:
    def __init__(self):
        pass


def imshow(title,im):
    """Decorator for OpenCV "imshow()" to handle images with transparency"""

    # Check we got np.uint8, 2-channel (grey + alpha) or 4-channel RGBA image
    if (im.dtype == np.uint8) and (len(im.shape)==3) and (im.shape[2] in set([2,4])):

        # Pick up the alpha channel and delete from original
        alpha = im[...,-1]/255.0
        im = np.delete(im, -1, -1)

        # Promote greyscale image to RGB to make coding simpler
        if len(im.shape) == 2:
            im = np.stack((im,im,im))

        h, w, _ = im.shape

        # Make a checkerboard background image same size, dark squares are grey(102), light squares are grey(152)
        f = lambda i, j: 102 + 50*((i+j)%2)
        bg = np.fromfunction(np.vectorize(f), (16,16)).astype(np.uint8)

        # Resize to square same length as longer side (so squares stay square), then trim
        if h>w:
            longer = h
        else:
            longer = w
        bg = cv2.resize(bg, (longer,longer), interpolation=cv2.INTER_NEAREST)
        # Trim to correct size
        bg = bg[:h,:w]

        # Blend, using result = alpha*overlay + (1-alpha)*background
        im = (alpha[...,None] * im + (1.0-alpha[...,None])*bg[...,None]).astype(np.uint8)

    cv2.imshow(title,im)

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

def visualise_segments():
    data_dir = "/home/aditya/real_output/45608/"
    camera_data = "/home/aditya/real_output/45608/camera_data.dat"
    log_reader = tricamera.LogReader(camera_data)
    orb = cv2.ORB_create()
    for observation in log_reader.data:

        # read images from raw data
        image60 = convert_image(observation.cameras[0].image, format="bgr")
        image180 = convert_image(observation.cameras[1].image, format="bgr")
        image300 = convert_image(observation.cameras[2].image, format="bgr")

        # read mask information
        mask60 = segment_image(image60)
        mask180 = segment_image(image180)
        mask300 = segment_image(image300)

        h, w, c = image60.shape


        # x60, y60 = np.where(mask60==255)
        x60, y60 = np.where(mask60==0)
        x180, y180 = np.where(mask180==0)
        x300, y300 = np.where(mask300==0)

        # mod60[x60, y60, :] = image60[x60, y60, :]
        image60[x60, y60, :] = 0
        image180[x180, y180, :] = 0
        image300[x300, y300, :] = 0

        gray60 = cv2.cvtColor(image60, cv2.COLOR_BGR2GRAY)
        gray180 = cv2.cvtColor(image180, cv2.COLOR_BGR2GRAY)
        gray300 = cv2.cvtColor(image300, cv2.COLOR_BGR2GRAY)

        ######################################################################
        #                Uncomment to visualise orb features                 #
        ######################################################################
        # gray60 = cv2.cvtColor(image60, cv2.COLOR_BGR2GRAY)

        # kp60, desc60 = orb.detectAndCompute(gray60, None)
        # # kp180 = sift.detect(gray180, None)
        # # kp300 = sift.detect(gray300, None)

        # img=cv2.drawKeypoints(gray60 ,
        #                     kp60 ,
        #                     image60)
        #                     #flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


        # cv2.imshow("camera60", img)
        ######################################################################
        #                        Uncomment till here                         #
        ######################################################################
        # cv2.imshow("camera60", image60)
        # cv2.imshow("camera180", image180)
        # cv2.imshow("camera300", image300)
        cv2.imshow("camera60", 255-gray60)
        cv2.imshow("camera180", 255-gray180)
        cv2.imshow("camera300", 255-gray300)
        if cv2.waitKey(33) == ord('q'):
            sys.exit()
    return

def visualise_blobs():
    data_dir = "/home/aditya/real_output/45608/"
    camera_data = "/home/aditya/real_output/45608/camera_data.dat"
    log_reader = tricamera.LogReader(camera_data)
    detector = cv2.SimpleBlobDetector()
    for observation in log_reader.data:

        # read images from raw data
        image60 = convert_image(observation.cameras[0].image, format="bgr")
        image180 = convert_image(observation.cameras[1].image, format="bgr")
        image300 = convert_image(observation.cameras[2].image, format="bgr")

        # read mask information
        mask60 = segment_image(image60)
        mask180 = segment_image(image180)
        mask300 = segment_image(image300)

        # apply blurring on the mask to remove any rogue holes within the
        # segment
        mask60_blur = cv2.medianBlur(mask60, 5)

        # convert original image to grayscale
        gray60 = cv2.cvtColor(image60, cv2.COLOR_BGR2GRAY)

        # get the negative of the image
        gray60_neg = 255-gray60

        # look in the mask where dice exist
        x60, y60 = np.where(mask60_blur==0)

        # make non dice pixels 0
        gray60_neg[x60, y60] = 0

        edge = cv2.Canny(gray60_neg, 100, 200)

        h, w, c = image60.shape

        ret, gray60_thresh = cv2.threshold(gray60_neg, 100, 255, cv2.THRESH_BINARY)
        gray60_adapThresh = cv2.adaptiveThreshold(gray60_neg, 255,
                                                  cv2.ADAPTIVE_THRESH_MEAN_C
                                                  , cv2.THRESH_BINARY, 5, 4)
        gray60_adapGaussThresh = cv2.adaptiveThreshold(gray60_neg, 255,
                                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, 5, 4)

        # keypoints = detector.detect(mask60)
        # mask60_key = cv2.drawKeypoints(mask60, keypoints, np.array([]),
        #                                (0,0,255),
        #                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("camera60", gray60_thresh)
        cv2.imshow("adaptive", gray60_adapThresh)
        cv2.imshow("adaptive_gauss", gray60_adapGaussThresh)
        cv2.imshow("dice", gray60_neg)
        cv2.imshow("edges", edge)
        if cv2.waitKey(33) == ord('q'):
            sys.exit()


    return

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
        # print("components in 60: {}".format(out60[0]))
        out180 = cv2.connectedComponentsWithStats(mask180, 4, cv2.CV_32S)
        # print("components in 180: {}".format(out180[0]))
        out300 = cv2.connectedComponentsWithStats(mask300, 4, cv2.CV_32S)
        # print("components in 300: {}".format(out300[0]))


        # show the connected component results
        # cv2.imshow("camera60", imshow_components(out60[1]))
        # cv2.imshow("camera180", imshow_components(out180[1]))
        # cv2.imshow("camera300", imshow_components(out300[1]))

        # cv2.imshow("cameramod60", mod60)
        # cv2.imshow("cameramod180", mod180)
        # cv2.imshow("cameramod300", mod300)
        # cv2.imshow("camera60", image60)
        # cv2.imshow("camera180", image180)
        # cv2.imshow("camera300", image300)
        # cv2.waitKey(100)
        imshow("cameramod60", mod60)
        imshow("cameramod180", mod180)
        imshow("cameramod300", mod300)
        # imshow("camera60", image60)
        # imshow("camera180", image180)
        # imshow("camera300", image300)
        if cv2.waitKey(33) == ord('q'):
            sys.exit()
    # open the segmentation maps
    # draw connected components on the segmentation maps
    # visualise

if __name__ == "__main__":
    # main()
    # visualise_segments()
    visualise_blobs()
