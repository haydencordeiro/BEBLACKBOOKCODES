

import cv2
import numpy as np
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
import time
import os
from flask_socketio import SocketIO, send
import socketio
import os
from pathlib import Path
import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sio = socketio.Client()
sio.connect('http://127.0.0.1:5000')
def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
    
    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    
    
def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    
    return (x, y)
    
face_model = get_face_detector()
landmark_model = get_landmark_model()

font = cv2.FONT_HERSHEY_SIMPLEX 
model_points = np.array([
                            (0.0, 0.0, 0.0),             
                            (0.0, -330.0, -65.0),     
                            (-225.0, 170.0, -135.0),
                            (225.0, 170.0, -135.0),
                            (-150.0, -150.0, -125.0),
                            (150.0, -150.0, -125.0)
                        ])


def main_function(image_path):

    img = cv2.imread(image_path)
    # ret, img = cap.read()
    size = img.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                         )
    faces = find_faces(img, face_model)
    for face in faces:
        marks = detect_marks(img, landmark_model, face)
        image_points = np.array([
                                marks[30],    
                                marks[8],    
                                marks[36],   
                                marks[45],   
                                marks[48],     
                                marks[54]      
                            ], dtype="double")
        dist_coeffs = np.zeros((4,1)) 
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
        
        

        
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
        for p in image_points:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        
        
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

        cv2.line(img, p1, p2, (0, 255, 255), 2)
        cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
        try:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            ang1 = int(math.degrees(math.atan(m)))
        except:
            print(":asdf")
            ang1 = 90
            
        try:
            m = (x2[1] - x1[1])/(x2[0] - x1[0])
            ang2 = int(math.degrees(math.atan(-1/m)))
        except:
            print(":asdf2")
            ang2 = 90
            
            # print('div by zero error')
        if ang1 >= 15:
            return ((abs(ang1))+(abs(ang2)))
            cv2.putText(img, 'down', (30, 30), font, 2, (255, 255, 128), 3)
        elif ang1 <= -15:
            return ((abs(ang1))+(abs(ang2)))
            cv2.putText(img, 'up', (30, 30), font, 2, (255, 255, 128), 3)
            
        if ang2 >= 15:
            return ((abs(ang1))+(abs(ang2)))
            cv2.putText(img, 'right', (90, 30), font, 2, (255, 255, 128), 3)
        elif ang2 <= -15:
            return ((abs(ang1))+(abs(ang2)))
            cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
        else:
            return 0
        cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
        cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
    return  "Ignore"



while True:
        it= os.listdir(r"D:\BEFINAL\datastream\orient")
        it.sort(key=lambda x:os.path.getctime("D:/BEFINAL/datastream/orient/"+ x))
        for entry in it:
            try:
                temp=Path(entry).name.split("-")
                person_id=temp[0]
                uid=temp[1][:-4]
                with open("D:/BEFINAL/datastream/orient"+"/"+entry, 'rb') as f:
                    check_chars = f.read()[-2:]
                if check_chars != b'\xff\xd9':
                    continue
                
                res=main_function("D:/BEFINAL/datastream/orient/"+entry)
                if res!=0 and res!="Ignore":
                    res/=120
                if res=="Ignore":
                    pass
                else:
                    sio.send(f"orient-{person_id}-{res}-{uid}")
                os.remove("D:/BEFINAL/datastream/orient/"+entry)
            except Exception as e:
                print(uid,e,"head_pose_estimation_image.py")
            




cv2.destroyAllWindows()
