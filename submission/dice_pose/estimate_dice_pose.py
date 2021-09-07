# some general imports
import numpy as np
from numpy.polynomial import polynomial as P
import cv2
import pathlib
import sys
from queue import Queue
import itertools
import typing
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import minimize

# trifinger imports
import trifinger_cameras.py_tricamera_types as tricamera
from trifinger_cameras.utils import convert_image
from trifinger_object_tracking.py_lightblue_segmenter import segment_image
from trifinger_simulation.camera import (
    load_camera_parameters,
    CameraParameters,
)

# optimisation related libs
import torch
import torch.nn as nn
from torch import optim as opt
from torch.autograd import Function

###############################################################################
#                     Some constants related to the arena                     #
###############################################################################

#: Radius of the arena in which target positions are sampled [m].
ARENA_RADIUS = 0.19

#: Number of dice in the arena
NUM_DICE = 25

#: Width of a die [m].
DIE_WIDTH = 0.022

#: Tolerance that is added to the target box width [m].
TOLERANCE = 0.003

#: Width of the target box in which the die has to be placed [m].
TARGET_WIDTH = DIE_WIDTH + TOLERANCE

#: Number of cells per row (one cell fits one die)
N_CELLS_PER_ROW = int(2 * ARENA_RADIUS / DIE_WIDTH)

FACE_CORNERS = (
    (0, 1, 2, 3),
    (4, 5, 1, 0),
    (5, 6, 2, 1),
    (7, 6, 2, 3),
    (4, 7, 3, 0),
    (4, 5, 6, 7),
)

###############################################################################
#                        End Constant decration block                         #
###############################################################################

# Helper types for type hints
Cell = typing.Tuple[int, int]
Position = typing.Sequence[float]
Goal = typing.Sequence[Position]


class DicePose:
    def __init__(self):
        pass

def draw_line(image, points):
    color = (0, 0, 255)
    # __import__('pudb').set_trace()
    cv2.line(image, (int(points[0][0][0]), int(points[0][0][1])),
             (int(points[1][0][0]), int(points[1][0][1])), color, 1)
    cv2.line(image, (int(points[0][0][0]), int(points[0][0][1])),
             (int(points[3][0][0]), int(points[3][0][1])), color, 1)

    cv2.line(image, (int(points[0][0][0]), int(points[0][0][1])),
             (int(points[4][0][0]), int(points[4][0][1])), color, 1)
    cv2.line(image, (int(points[1][0][0]), int(points[1][0][1])),
             (int(points[2][0][0]), int(points[2][0][1])), color, 1)
    cv2.line(image, (int(points[1][0][0]), int(points[1][0][1])),
             (int(points[5][0][0]), int(points[5][0][1])), color, 1)
    cv2.line(image, (int(points[2][0][0]), int(points[2][0][1])),
             (int(points[3][0][0]), int(points[3][0][1])), color, 1)
    cv2.line(image, (int(points[2][0][0]), int(points[2][0][1])),
             (int(points[6][0][0]), int(points[6][0][1])), color, 1)

    cv2.line(image, (int(points[3][0][0]), int(points[3][0][1])),
             (int(points[7][0][0]), int(points[7][0][1])), color, 1)
    cv2.line(image, (int(points[4][0][0]), int(points[4][0][1])),
             (int(points[7][0][0]), int(points[7][0][1])), color, 1)

    cv2.line(image, (int(points[4][0][0]), int(points[4][0][1])),
             (int(points[5][0][0]), int(points[5][0][1])), color, 1)
    cv2.line(image, (int(points[5][0][0]), int(points[5][0][1])),
             (int(points[6][0][0]), int(points[6][0][1])), color, 1)
    cv2.line(image, (int(points[6][0][0]), int(points[6][0][1])),
             (int(points[7][0][0]), int(points[7][0][1])), color, 1)

    # cv2.line(image,(points[1][0][0]), (0, 0, 255), 6)
    # cv2.line(image,(), (0, 0, 255), 6)
    # cv2.line(image,(), (0, 0, 255), 6)
    # cv2.line(image,(), (0, 0, 255), 6)
    # cv2.line(image,(), (0, 0, 255), 6)
    # cv2.line(image,(), (0, 0, 255), 6)
    # cv2.line(image,(), (0, 0, 255), 6)
    # cv2.line(image,(), (0, 0, 255), 6)
    # cv2.line(image,(), (0, 0, 255), 6)
    # cv2.line(image,(), (0, 0, 255), 6)
    # cv2.line(image,(), (0, 0, 255), 6)
    # cv2.line(image,(), (0, 0, 255), 6)

    return image

def _get_cell_corners_3d(
    pos: Position,
) -> np.ndarray:
    """Get 3d positions of the corners of the cell at the given position."""
    d = DIE_WIDTH / 2
    nppos = np.asarray(pos)

    # order of the corners is the same as in the cube model of the
    # trifinger_object_tracking package
    # people.tue.mpg.de/mpi-is-software/robotfingers/docs/trifinger_object_tracking/doc/cube_model.html
    return np.array(
        (
            nppos + (d, -d, d),
            nppos + (d, d, d),
            nppos + (-d, d, d),
            nppos + (-d, -d, d),
            nppos + (d, -d, -d),
            nppos + (d, d, -d),
            nppos + (-d, d, -d),
            nppos + (-d, -d, -d),
        )
    )

class ProjectCube:
    def __init__(self):
        # Set camera parameters as used in simulation
        # view 1
        pose_60 = (
            np.array(
                (
                    -0.6854993104934692,
                    -0.5678349733352661,
                    0.45569100975990295,
                    0.0,
                    0.7280372381210327,
                    -0.5408401489257812,
                    0.4212528169155121,
                    0.0,
                    0.007253906223922968,
                    0.6205285787582397,
                    0.7841504216194153,
                    0.0,
                    -0.01089033205062151,
                    0.014668643474578857,
                    -0.5458434820175171,
                    1.0,
                )
            )
            .reshape(4, 4)
            .T
        )
        # view 2
        pose_180 = (
            np.array(
                (
                    0.999718189239502,
                    0.02238837257027626,
                    0.007906466722488403,
                    0.0,
                    -0.01519287470728159,
                    0.8590874671936035,
                    -0.5116034150123596,
                    0.0,
                    -0.01824631541967392,
                    0.5113391280174255,
                    0.8591853380203247,
                    0.0,
                    -0.000687665306031704,
                    0.01029178500175476,
                    -0.5366422533988953,
                    1.0,
                )
            )
            .reshape(4, 4)
            .T
        )
        # view 3
        pose_300 = (
            np.array(
                (
                    -0.7053901553153992,
                    0.5480064153671265,
                    -0.44957074522972107,
                    0.0,
                    -0.7086654901504517,
                    -0.5320233702659607,
                    0.4634052813053131,
                    0.0,
                    0.014766914770007133,
                    0.6454768180847168,
                    0.7636371850967407,
                    0.0,
                    -0.0019663232378661633,
                    0.0145435631275177,
                    -0.5285998582839966,
                    1.0,
                )
            )
            .reshape(4, 4)
            .T
        )
        # proj
        pb_proj = (
            np.array(
                (
                    2.0503036975860596,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0503036975860596,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0000200271606445,
                    -1.0,
                    0.0,
                    0.0,
                    -0.002000020118430257,
                    0.0,
                )
            )
            .reshape(4, 4)
            .T
        )
        width = 270
        height = 270
        x_scale = pb_proj[0, 0]
        y_scale = pb_proj[1, 1]
        c_x = width / 2
        c_y = height / 2
        f_x = x_scale * c_x
        f_y = y_scale * c_y
        camera_matrix = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 0]])

        dist = (0, 0, 0, 0, 0)

        self.camera_params = (
            CameraParameters(
                "camera60", width, height, camera_matrix, dist, pose_60
            ),
            CameraParameters(
                "camera180", width, height, camera_matrix, dist, pose_180
            ),
            CameraParameters(
                "camera300", width, height, camera_matrix, dist, pose_300
            ),
        )
        pass

    def projectPoints(self, pos: Position):
        """
        Project the given 3d position onto the camera's perspective
        individually
        """


        # get camera position and orientation separately
        tvec = self.camera_params[0].tf_world_to_camera[:3, 3] # translation vector
        rmat = self.camera_params[0].tf_world_to_camera[:3, :3] # rotation matrix
        rvec = Rotation.from_matrix(rmat).as_rotvec() # convert to rotation vector

        # retrieve corners of the imaginary cube
        corners = _get_cell_corners_3d(pos)

        # project corner points into the image
        projected_corners, _ = cv2.projectPoints(
            corners,
            rvec,
            tvec,
            self.camera_params[0].camera_matrix,
            self.camera_params[0].distortion_coefficients,
        )

        return projected_corners


###############################################################################
#               Exploration funtions for dice pose construction               #
###############################################################################

def render_cube(project_cube, pos, image):
    """render_cube.
    Projects a 3d cube onto the image, and then renders it on the image
    provided

    Args:
        project_cube: instance of ProjectCube class that is used to project the
        cube
        pos: position in the real world where the cube needs to be rendered
        image: image to render cube on
    """
    point_numpy = np.asarray([pos[0], pos[1], 0.05])
    points = project_cube.projectPoints(point_numpy)
    for point in points:
        cv2.circle(image, (int(point[0][0]), int(point[0][1])), 0, (0, 0, 255))

    draw_line(image, points)
    return points

class HausdorffOptim():
    def __init__(self, polygon, project_cube, image):
        self.polygon = polygon
        self.project_cube = project_cube
        self.image = image
        self.result = []

    def hausdorff_loss(self, x):
        point_numpy = np.asarray([x[0], x[1], 0.05])
        points = self.project_cube.projectPoints(point_numpy)
        return hausdorf_distance(np.squeeze(self.polygon, axis=1),
                                 np.squeeze(points, axis=1))

    def fit(self):
        options={'atol':1, 'disp':True}
        pos = [0.0, 0.0]
        result = minimize(self.hausdorff_loss, pos, method="nelder-mead",
                          options=options, callback=self.visualise)
        return result.x

    def visualise(self, xk):
        cv2.drawContours(self.image, [self.polygon], -1, (255, 255, 0))
        _ = render_cube(self.project_cube, xk, self.image)

        cv2.imshow("optimising", self.image)
        cv2.waitKey(33)




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

def draw_points(image, points):
    for i in range(0, np.shape(points)[0]):
        cv2.circle(image, (int(points[i][0]), int(points[i][1])), 1, (0, 255, 0),
                   -1)

def draw_point(image, point):
    cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)


def pointPolygonCheck(polygon, point):
    dis = cv2.pointPolygonTest(polygon, point, True)
    return dis

def whichPolygon(polygon_list, centroid):
    for polygon in polygon_list:
        # __import__('pudb').set_trace()
        dis = pointPolygonCheck(polygon, centroid)
        if dis>=0:
            return polygon

    return None

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

def merge_lines(lines):
    clusters =  []
    idx = []
    total_lines = len(lines)
    distance_threshold = 7
    #if total_lines < 30:
    #    distance_threshold = 20
    #elif total_lines <75:
    #    distance_threshold = 15
    #elif total_lines<120:
    #    distance_threshold = 10
    #else:
    #    distance_threshold = 7
    for i,line in enumerate(lines):
        x1,y1,x2,y2 = line
        if [x1,y1,x2,y2] in idx:
            continue
        parameters = P.polyfit((x1, x2),(y1, y2), 1)
        slope = parameters[0]#(y2-y1)/(x2-x1+0.001)
        intercept = parameters[1]#((y2+y1) - slope *(x2+x1))/2
        a = -slope
        b = 1
        c = -intercept
        d = np.sqrt(a**2+b**2)
    cluster = [line]
    for d_line in lines[i+1:]:
        x,y,xo,yo= d_line
        mid_x = (x+xo)/2
        mid_y = (y+yo)/2
        distance = np.abs(a*mid_x+b*mid_y+c)/d
        if distance < distance_threshold:
            cluster.append(d_line)
            idx.append(d_line.tolist())
    clusters.append(np.array(cluster))
    merged_lines = [np.mean(cluster, axis=0) for cluster in clusters]
    # print(clusters)
    # print(merged_lines)
    return merged_lines

def visualise_blobs():
    data_dir = "/home/aditya/real_output/45615/"
    camera_data = "/home/aditya/real_output/45615/camera_data.dat"
    log_reader = tricamera.LogReader(camera_data)
    detector = cv2.SimpleBlobDetector()

    # initialise queue to store points from n consecutive frames
    n_line = 10
    n_con = 1
    line_q = Queue(maxsize=n_line)
    contour_q = Queue(maxsize=n_con)

    project_cube = ProjectCube()

    for observation in log_reader.data:

        # read images from raw data
        image60 = convert_image(observation.cameras[0].image, format="bgr")
        image180 = convert_image(observation.cameras[1].image, format="bgr")
        image300 = convert_image(observation.cameras[2].image, format="bgr")

        # read mask information
        mask60 = segment_image(image60)
        mask180 = segment_image(image180)
        mask300 = segment_image(image300)

        ######################################################################
        #                   compute connected componenets                    #
        ######################################################################

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
        # __import__('pudb').set_trace()
        # print("components in 300: {}".format(out300[0]))
        ######################################################################
        #                  end of connected component blobs                  #
        ######################################################################
        

        # apply blurring on the mask to remove any rogue holes within the
        # segment
        mask60_blur = cv2.medianBlur(mask60, 5)

        # convert original image to grayscale
        gray60 = cv2.cvtColor(image60, cv2.COLOR_BGR2GRAY)

        # copy of the original image
        image60_cp = np.copy(image60)

        # make a copy for overlaying contours
        image_cor = np.copy(image60)

        # make a copy for overlaying approximated polygons
        image_poly = np.copy(image60)

        # make a copy of the image to project an imaginary cube onto the image
        image_proj = np.copy(image60)

        # get the negative of the image
        gray60_neg = 255-gray60

        # look in the mask where dice exist
        x60, y60 = np.where(mask60_blur==0)

        # make non dice pixels 0
        gray60_neg[x60, y60] = 0

        # image_cor[x60, y60, :] = 0

        # detect edges
        edge = cv2.Canny(gray60_neg, 200, 250)

        # if not edge_q.full():
        #     edge_q.put(edge)
        # else:
        #     edge_q.get()
        #     edge_q.put(edge)

        # cum_edges = list(itertools.chain(*list(edge_q.queue)))


        # edge = cv2.convertScaleAbs
        # kernel = np.ones((1,1), np.uint8)
        # edge = cv2.dilate(edge, kernel, iterations=1)

        ######################################################################
        #               Uncomment to try out corner detection                #
        ######################################################################
        # detect corners
        # corners = cv2.preCornerDetect(gray60_neg, 5)
        # corners = cv2.preCornerDetect(edge, 5)
        # harr_corners = cv2.cornerHarris(edge, 5, 5, 0.1)

        ######################################################################
        #              uncomment till here for corner detection              #
        ######################################################################

        edges_cp = np.copy(edge)

        # contouring code
        contours, h = cv2.findContours(np.asarray(edge), cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)
        if not contour_q.full():
            contour_q.put(contours)
        else:
            contour_q.get()
            contour_q.put(contours)

        cum_contours = list(itertools.chain(*list(contour_q.queue)))
        cv2.drawContours(image_cor, cum_contours, -1, (0, 255, 0))

        ######################################################################
        #                        Approximate Polygons                        #
        ######################################################################

        # approximate polygons for each contour
        polygons = []
        for cnt in cum_contours:
            poly = cv2.approxPolyDP(cnt, 3, True)
            polygons.append(poly)

        cv2.drawContours(image_poly, polygons, -1, (255, 255, 0))

        ######################################################################
        #                       Hough line Experiments                       #
        ######################################################################
        # find the probabilistic hough line transform
        lines = cv2.HoughLinesP(edges_cp, 1, np.pi/180, 5)

        # skip canny
        # lines = cv2.HoughLinesP(gray60_neg, 1, np.pi/180, 50)


        if not line_q.full():
            line_q.put(lines)
        else:
            line_q.get()
            line_q.put(lines)

        # merge the lines above
        # lines = np.squeeze(lines, axis=1)
        # m_lines = merge_lines(lines)

        cum_lines = list(itertools.chain(*list(line_q.queue)))

        if cum_lines is not None:
            for i in range(0, len(cum_lines)):
                l = cum_lines[i][0]
                cv2.line(image60_cp, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1,
                         cv2.LINE_AA)
                # x1, y1, x2, y2 = m_lines[i].astype(int)
                # cv2.line(image60_cp, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # find contours
        # contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE,
        #                                        cv2.CHAIN_APPROX_SIMPLE)
        ######################################################################
        #                     uncomment for contour code                     #
        ######################################################################
        # contours, hierarchy = cv2.findContours(gray60_neg, cv2.RETR_TREE,
        #                                        cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(image60, contours, -1, (0, 255, 0), 3)

        ######################################################################
        #                        uncomment till here                         #
        ######################################################################

        # h, w, c = image60.shape

        ret, gray60_thresh = cv2.threshold(gray60_neg, 100, 255, cv2.THRESH_BINARY)
        # gray60_adapThresh = cv2.adaptiveThreshold(gray60_neg, 255,
        #                                           cv2.ADAPTIVE_THRESH_MEAN_C
        #                                           , cv2.THRESH_BINARY, 5, 4)
        # gray60_adapGaussThresh = cv2.adaptiveThreshold(gray60_neg, 255,
        #                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                                cv2.THRESH_BINARY, 5, 4)

        # keypoints = detector.detect(mask60)
        # mask60_key = cv2.drawKeypoints(mask60, keypoints, np.array([]),
        #                                (0,0,255),
        #                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ######################################################################
        #               section to project 3d cube onto image                #
        ######################################################################
        pos = (0.0, 0.0, 0.01)
        # points = project_cube.projectPoints(pos)
        # for point in points:
        #     # __import__('pudb').set_trace()
        #     cv2.circle(image_proj, (int(point[0][0]), int(point[0][1])), 0, (0,0,255), -1)

        render_cube(project_cube, pos, image_proj)
        # draw_line(image_proj, points)

        cv2.imshow("camera60", gray60_thresh)
        cv2.imshow("proj cube", image_proj)
        # cv2.imshow("adaptive", gray60_adapThresh)
        # cv2.imshow("adaptive_gauss", gray60_adapGaussThresh)
        cv2.imshow("dice", image60)
        cv2.imshow("edges", edge)
        cv2.imshow("hough lines", image60_cp)
        cv2.imshow("contours", image_cor)
        cv2.imshow("approx polygons", image_poly)
        # cv2.imshow("camera60 components", imshow_components(out60[1] == 16))

        image_temp = imshow_components(out60[1])
        render_cube(project_cube, pos, image_temp)
        cv2.imshow("camera60 components", image_temp)
        # cv2.imshow("corners", corners)
        # cv2.imshow("harris corners", harr_corners)
        # cv2.imshow("contoured image", contoured_image)
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

def estimate_pose():

    # initialise data reading
    data_dir = "/home/aditya/real_output/45608/"
    camera_data = "/home/aditya/real_output/45608/camera_data.dat"
    log_reader = tricamera.LogReader(camera_data)

    # cube projection instance
    project_cube = ProjectCube()

    # maintain a queue for cobntours
    n_con = 1
    contour_q = Queue(maxsize=n_con)

    for observation in log_reader.data:

        # read images from raw data
        image60 = convert_image(observation.cameras[0].image, format="bgr")
        image180 = convert_image(observation.cameras[1].image, format="bgr")
        image300 = convert_image(observation.cameras[2].image, format="bgr")

        # read mask information
        mask60 = segment_image(image60)
        mask180 = segment_image(image180)
        mask300 = segment_image(image300)

        ######################################################################
        #                   compute connected componenets                    #
        ######################################################################

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
        # __import__('pudb').set_trace()
        # print("components in 300: {}".format(out300[0]))
        ######################################################################
        #                  end of connected component blobs                  #
        ######################################################################

        # apply blurring on the mask to remove any rogue holes within the
        # segment
        mask60_blur = cv2.medianBlur(mask60, 5)

        # convert original image to grayscale
        gray60 = cv2.cvtColor(image60, cv2.COLOR_BGR2GRAY)

        # get the negative of grayscale image
        gray60_neg = 255-gray60

        # look in the mask where dice exists
        x60, y60 = np.where(mask60_blur==0)

        # make non dice pixels 0
        gray60_neg[x60, y60] = 0

        # detect edges
        edge = cv2.Canny(gray60_neg, 200, 250)

        # get a copy of the edges
        edges_cp = np.copy(edge)

        # get contours
        contours, h = cv2.findContours(np.asarray(edge), cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)
        # update the contour queue
        if not contour_q.full():
            contour_q.put(contours)
        else:
            contour_q.get()
            contour_q.put(contours)

        # maintain a list of cumulative contours
        cum_contours = list(itertools.chain(*list(contour_q.queue)))

        # approximate polygons for each contour
        polygons = []
        for cnt in cum_contours:
            poly = cv2.approxPolyDP(cnt, 3, True)
            if poly.shape[0] >= 4:
                polygons.append(poly)


        # now we have everything we need to construct pose of dice
        # let's visualise it

        image_temp = imshow_components(out60[1] == 17)
        image_optim = np.copy(image_temp)
        target_poly = whichPolygon(polygons, out60[3][17])
        if target_poly is None:
            raise ValueError("There is no polygon corresponding to this\
                             centroid")
        cv2.drawContours(image_temp, [target_poly], -1, (255, 255, 0))
        # cv2.drawContours(image_temp, polygons, -1, (255, 255, 0))
        pos = [0.0, 0.0]
        projected_points = render_cube(project_cube, pos, image_optim)

        # you now have two sets of points
        # let's optimise the hausdorff distance and visualise
        # calculate loss
        # backward pass
        # calculate gradients
        # update
        # visualise
        optimiser = HausdorffOptim(target_poly, project_cube, image_optim)
        optimiser.fit()
        sys.exit()




        __import__('pudb').set_trace()
        # draw_points(image_temp, out60[3][16])
        draw_point(image_temp, out60[3][17])
        cv2.imshow("camera60 components", image_temp)
        key = cv2.waitKey(0)
        if key == ord('q'):
            sys.exit()
        elif key == ord('c'):
            continue



def hausdorf_distance(set1, set2):
    """hausdorf_distance.
    computes hausdorff distance between two sets of points

    Args:
        set1: numpy array of the order (M, N)
        set2: numpy array of the order (O, N)
    """
    # __import__('pudb').set_trace()
    return max(directed_hausdorff(set1, set2)[0], directed_hausdorff(set2,
                                                                     set1)[0])






if __name__ == "__main__":
    # main()
    # visualise_segments()
    # visualise_blobs()
    estimate_pose()
