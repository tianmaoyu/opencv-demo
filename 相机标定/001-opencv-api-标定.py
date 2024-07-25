import cv2 as cv
import numpy as np
import glob
import xml.etree.ElementTree as ET


class CameraCalibrator(object):
    def __init__(self, image_size: tuple):
        super(CameraCalibrator, self).__init__()
        self.image_size = image_size
        self.matrix = np.zeros((3, 3), np.float)
        self.new_camera_matrix = np.zeros((3, 3), np.float)
        self.dist = np.zeros((1, 5))
        self.roi = np.zeros(4, np.int)

    def cal_real_corner(self, corner_height, corner_width, square_size):
        # corner世界坐标系坐标：chessboard平面为z=0的平面，左下角inner corners作为原点
        obj_corner = np.zeros([corner_height * corner_width, 3], np.float32)
        obj_corner[:, :2] = np.mgrid[0:corner_height, 0:corner_width].T.reshape(-1, 2)  # (w*h)*2
        return obj_corner * square_size

    def calibration(self, corner_height: int, corner_width: int, square_size: float):
        file_names = glob.glob('./chess/*.JPG')
        objs_corner = []
        imgs_corner = []
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        obj_corner = self.cal_real_corner(corner_height, corner_width, square_size)  # 3d point in real world space
        for file_name in file_names:
            # read image
            chess_img = cv.imread(file_name)
            assert (chess_img.shape[0] == self.image_size[0] and chess_img.shape[1] == self.image_size[1]), \
                "Image size does not match the given value {}. {}".format(self.image_size,chess_img.shape)
            # to gray
            gray = cv.cvtColor(chess_img, cv.COLOR_BGR2GRAY)
            # find chessboard corners
            ret, img_corners = cv.findChessboardCorners(gray, (corner_height, corner_width))

            # append to img_corners
            if ret:
                objs_corner.append(obj_corner)
                # 优化角点检测结果达到亚像素级
                img_corners = cv.cornerSubPix(gray, img_corners, winSize=(square_size // 2, square_size // 2),
                                              zeroZone=(-1, -1), criteria=criteria)
                imgs_corner.append(img_corners)
                '''
                # Draw and display the corners
                cv.drawChessboardCorners(gray, (corner_height,corner_width), img_corners, ret)
                cv.imshow('img', gray)
                cv.waitKey(500)
                cv.destroyAllWindows()
                '''
            else:
                print("Fail to find corners in {}.".format(file_name))

        # calibration
        ret, self.matrix, self.dist, rvecs, tveces = cv.calibrateCamera(objs_corner, imgs_corner, self.image_size, None,
                                                                        None)
        # 根据比例因子alpha返回相应的新的相机内参矩阵
        self.new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(self.matrix, self.dist, self.image_size, alpha=1)
        self.roi = np.array(roi)
        return ret

    def save_params(self, save_path='camera_params.xml'):
        root = ET.Element('root')
        tree = ET.ElementTree(root)

        comment = ET.Element('about')
        comment.set('author', 'XXXX')
        comment.set('github', 'XXXXX')
        root.append(comment)
        mat_node = ET.Element('camera_matrix')
        root.append(mat_node)
        for i, elem in enumerate(self.matrix.flatten()):
            child = ET.Element('data{}'.format(i))
            child.text = str(elem)
            mat_node.append(child)

        new_node = ET.Element('new_camera_matrix')
        root.append(new_node)
        for i, elem in enumerate(self.new_camera_matrix.flatten()):
            child = ET.Element('data{}'.format(i))
            child.text = str(elem)
            new_node.append(child)

        dist_node = ET.Element('camera_distortion')
        root.append(dist_node)
        for i, elem in enumerate(self.dist.flatten()):
            child = ET.Element('data{}'.format(i))
            child.text = str(elem)
            dist_node.append(child)

        roi_node = ET.Element('roi')
        root.append(roi_node)
        for i, elem in enumerate(self.roi):
            child = ET.Element('data{}'.format(i))
            child.text = str(elem)
            roi_node.append(child)

        tree.write(save_path, 'UTF-8')
        print("Saved params in {}.".format(save_path))


if __name__ == '__main__':
    # width*height of chessboard corner
    corner = [11, 8]  # Number of inner corners per a item row and column
    square = 60  # size of chessboard square(Necessary when calibrating)
    image_size = (4032,3024)
    calibrator = CameraCalibrator(image_size)
    if calibrator.calibration(corner[1], corner[0], square):
        calibrator.save_params()
    else:
        print("Calibration failed.")

    exit(1)