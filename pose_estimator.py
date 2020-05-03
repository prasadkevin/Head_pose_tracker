"""Estimate head pose according to the facial landmarks"""
import cv2
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt 
from statistics import mean 

pitch_list = []
yaw_list = []
roll_list = []

class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, img_size=(480, 640)):
        self.size = img_size

        # 3D model points.
        # for shape of the data ..structure to 68 get 68 points
        
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Mouth left corner
            (150.0, -150.0, -125.0)      # Mouth right corner
        ]) / 4.5

        self.model_points_68 = self._get_full_model_points()
        
#        print('model_points_68',self.model_points_68)

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
#        self.camera_matrix = np.array(
#            [[self.focal_length, 0, self.camera_center[0]],
#             [0, self.focal_length, self.camera_center[1]],
#             [0, 0, 1]], dtype="double")

    
        self.camera_matrix = np.array(
            [[1, 0, 2],
             [0, 212, 65],
             [3, 3, 3]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])
        # self.r_vec = None
        # self.t_vec = None

    def _get_full_model_points(self, filename='assets/model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1

        return model_points

    def show_3d_model(self):
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D
        fig = pyplot.figure()
        ax = Axes3D(fig)

        x = self.model_points_68[:, 0]
        y = self.model_points_68[:, 1]
        z = self.model_points_68[:, 2]

        ax.scatter(x, y, z)
        ax.axis('square')
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        pyplot.show()
#
    def solve_pose(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        assert image_points.shape[0] == self.model_points_68.shape[0], "3D points and 2D points should be of same number."
#        print('self.model_points_68.shape[0]',self.model_points_68.shape[0])
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeefs)

        (success, rotation_vector, translation_vector) = cv2.solvePnP(
             self.model_points,
             image_points,
             self.camera_matrix,
             self.dist_coeefs,
             rvec=self.r_vec,
             tvec=self.t_vec,
             useExtrinsicGuess=True)
        
        return (rotation_vector, translation_vector)

    def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))
        
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(point_2d[8]), color, line_width, cv2.LINE_AA)
        
        r_mat, _ = cv2.Rodrigues(rotation_vector)
        p_mat = np.hstack((r_mat, np.array([[0], [0], [0]])))
        _, _, _, _, _, _, u_angle = cv2.decomposeProjectionMatrix(p_mat)
        pitch, yaw, roll = u_angle.flatten()
   
        pitch_list.append(pitch)
        yaw_list.append(yaw)
        roll_list.append(roll)
        
        pitch_str = ''
        yaw_str = ''
        roll_str= ''
        i = 15
       
########################### for pitch
        if len(pitch_list) > i:
            pitch_avg = np.mean(pitch_list[-i]) 
            if pitch_avg >= 8 :
                pitch_str = 'up'
            elif round(pitch_avg) in range(-8,8):
                pitch_str = 'straight'
            elif pitch_avg <= -8 :
                pitch_str = 'down'
            pitch_list.pop(0)
            print('pitch >>>>>>>>>>>>>>>>   ' + str(pitch_avg))

########################### for yaw         
        if len(yaw_list) > i:
            yaw_avg = np.mean(yaw_list[-i]) 
            if yaw_avg >= 8:
                yaw_str = 'right'
            elif round(yaw_avg) in range(-8,8):
                yaw_str = 'straight'
            elif yaw_avg <= -8:
                yaw_str = 'left'
            yaw_list.pop(0)    
            print('yaw >>>>>>>>>>>>>>>>   ' + str(yaw_avg))

########################### for roll
            
        if len(roll_list) > i:
            roll_avg = np.mean(roll_list[-i])  
                    
            if -165 <= round(roll_avg) <= -130:
                roll_str ='bending_right'
            elif round(roll_avg) in range(130,165):
                roll_str ='bending_left'
            elif abs(round(roll_avg ))in  range(165,180):
                roll_str = 'straight' 
            
            roll_list.pop(0)
            print('roll >>>>>>>>>>>>>>>>   '+ str(roll_avg))

        looking_dict = {'pitch':pitch_str, 'yaw': yaw_str, 'roll' : roll_str}
        
         
        if all(value == 'straight' for value in looking_dict.values()) == True:
            return 'straight' 
        
        elif  roll_str == 'bending_right' and  (pitch_str == 'straight' or pitch_str == 'up' or pitch_str == 'down') :
            return 'bending_right'
        elif roll_str == 'bending_left' and  (pitch_str == 'straight' or pitch_str == 'up' or pitch_str == 'down'):
            return 'bending_left'
        elif pitch_str == 'up' and  roll_str == 'straight':
            return 'up'
        elif pitch_str == 'down' and  roll_str == 'straight':
            return 'down'
        elif yaw_str == 'left'and roll_str == 'straight' and pitch_str == 'straight':
            return 'left'
        elif yaw_str == 'right' and roll_str == 'straight' and pitch_str == 'straight':
            return 'right'
        
        return ''
    
#    def draw_axis(self, img, R, t):
#        points = np.float32(
#            [[30, 0, 0], [0, 30, 0], [0, 0, 30], [0, 0, 0]]).reshape(-1, 3)
#
#        axisPoints, _ = cv2.projectPoints(
#            points, R, t, self.camera_matrix, self.dist_coeefs)
#
#        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
#            axisPoints[0].ravel()), (255, 0, 0), 3)
#        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
#            axisPoints[1].ravel()), (0, 255, 0), 3)
#        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
#            axisPoints[2].ravel()), (0, 0, 255), 3)

    def draw_axes(self, img, R, t):
        img	= cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeefs, R, t, 30)
        
        print('self.camera_matrix',self.camera_matrix)

#        print('self.dist_coeefs',self.dist_coeefs)
        


    def get_pose_marks(self, marks):
        """Get marks ready for pose estimation from 68 marks"""
        pose_marks = []
        pose_marks.append(marks[30])    # Nose tip
        pose_marks.append(marks[8])     # Chin
        pose_marks.append(marks[36])    # Left eye left corner
        pose_marks.append(marks[45])    # Right eye right corner
        pose_marks.append(marks[48])    # Mouth left corner
        pose_marks.append(marks[54])    # Mouth right corner
        return pose_marks