import rospy
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
import numpy as np

def init_camera():
    cap = cv2.VideoCapture(-1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

rospy.init_node('camera_stream_publisher')

# Publishers for images and ArUco pose
image_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
undistorted_image_pub = rospy.Publisher("/camera/undistorted_image_raw", Image, queue_size=10)
pose_pub = rospy.Publisher("/camera/aruco_pose", Pose, queue_size=10)

bridge = CvBridge()

# Camera matrix (K) and distortion coefficients (D) from calibration
K = np.array([[229.53214543, 0, 322.49195729],
              [0, 229.39758169, 228.03407764],
              [0, 0, 1]])
D = np.array([[-0.13428436, 0.31066182, -0.35867337, 0.1338705]])

# Define ArUco parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
aruco_params = aruco.DetectorParameters_create()
marker_length = 0.18  # Marker size in meters (100mm)

cap = init_camera()

while not rospy.is_shutdown():
    ret, frame = cap.read()

    if not ret:
        print("Lost connection to camera, attempting to reconnect...")
        cv2.destroyAllWindows()
        cap.release()
        cap = init_camera()
        continue

    h, w = frame.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3))
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # ArUco detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # Pose estimation and publishing for all markers
    if ids is not None:
        for i in range(len(ids)):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i:i+1], marker_length, K, D)
            pose_msg = Pose()
            pose_msg.position.x = tvec[0][0][0]
            pose_msg.position.y = tvec[0][0][1]
            pose_msg.position.z = tvec[0][0][2]
            # Convert rotation vector to quaternion (optional based on your requirement)
            # ...
            pose_pub.publish(pose_msg)
        aruco.drawDetectedMarkers(frame, corners, ids)
        # aruco.drawAxis(frame, K, D, rvec, tvec, 0.1)    
    
    # Publish both original (distorted) and undistorted frames
    image_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
    image_pub.publish(image_msg)
    undistorted_image_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
    undistorted_image_pub.publish(undistorted_image_msg)

    # Optionally display the images
    # cv2.imshow('Distorted Frame', frame)
    # cv2.imshow('Undistorted Frame', undistorted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
