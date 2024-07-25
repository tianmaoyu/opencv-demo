import os

import cv2 as cv
import numpy as np
import glob
import xml.etree.ElementTree as ET


class CameraCalibrator(object):
    def __init__(self, image_size: tuple):
        super(CameraCalibrator, self).__init__()
        self.image_size = image_size
        self.matrix = np.zeros((3, 3), np.float64)
        self.new_camera_matrix = np.zeros((3, 3), np.float64)
        self.dist = np.zeros((1, 5))
        self.roi = np.zeros(4, np.int32)
        self.load_params()

    def load_params(self, param_file: str = 'camera_params.xml'):
        if not os.path.exists(param_file):
            print("File {} does not exist.", format(param_file))
            exit(-1)
        tree = ET.parse(param_file)
        root = tree.getroot()
        mat_data = root.find('camera_matrix')
        matrix = dict()
        if mat_data:
            for data in mat_data.iter():
                matrix[data.tag] = data.text
            for i in range(9):
                self.matrix[i // 3][i % 3] = float(matrix['data{}'.format(i)])
        else:
            print('No element named camera_matrix was found in {}'.format(param_file))

        new_camera_matrix = dict()
        new_data = root.find('new_camera_matrix')
        if new_data:
            for data in new_data.iter():
                new_camera_matrix[data.tag] = data.text
            for i in range(9):
                self.new_camera_matrix[i // 3][i % 3] = float(new_camera_matrix['data{}'.format(i)])
        else:
            print('No element named new_camera_matrix was found in {}'.format(param_file))

        dist = dict()
        dist_data = root.find('camera_distortion')
        if dist_data:
            for data in dist_data.iter():
                dist[data.tag] = data.text
            for i in range(5):
                self.dist[0][i] = float(dist['data{}'.format(i)])
        else:
            print('No element named camera_distortion was found in {}'.format(param_file))

        roi = dict()
        roi_data = root.find('roi')
        if roi_data:
            for data in roi_data.iter():
                roi[data.tag] = data.text
            for i in range(4):
                self.roi[i] = int(roi['data{}'.format(i)])
        else:
            print('No element named roi was found in {}'.format(param_file))

    def rectify_image(self, img):
        if not isinstance(img, np.ndarray):
            AssertionError("Image type '{}' is not numpy.ndarray.".format(type(img)))
        dst = cv.undistort(img, self.matrix, self.dist, self.new_camera_matrix)  # undistort
        return dst
        # crop the image
        x, y, w, h = self.roi
        dst = dst[y:y + h, x:x + w]
        dst = cv.resize(dst, (img.shape[0], img.shape[1]))
        return dst


if __name__ == '__main__':
    # width*height of chessboard corner
    corner = [11, 8]  # Number of inner corners per a item row and column
    square = 60  # size of chessboard square(Necessary when calibrating)
    image_size = (4032,3024)
    calibrator = CameraCalibrator(image_size)

    img = cv.imread('./chess/2.jpg')
    img_dst = calibrator.rectify_image(img)
    cv.imwrite("./output/2.jpg",img_dst)
    exit(1)
