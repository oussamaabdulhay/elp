import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np

def init_camera():
    cap = cv2.VideoCapture(-1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

rospy.init_node('camera_stream_publisher')

# Publishers for both distorted and undistorted images
image_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
undistorted_image_pub = rospy.Publisher("/camera/undistorted_image_raw", Image, queue_size=10)

bridge = CvBridge()

# Camera matrix (K) and distortion coefficients (D) from calibration
K = np.array([[229.53214543, 0, 322.49195729],
              [0, 229.39758169, 228.03407764],
              [0, 0, 1]])
D = np.array([[-0.13428436, 0.31066182, -0.35867337, 0.1338705]])

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
    # New optimal camera matrix for undistortion
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3))

    # Undistort the frame
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Publish both original (distorted) and undistorted frames
    image_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
    image_msg.header.stamp = rospy.get_rostime()
    image_pub.publish(image_msg)

    undistorted_image_msg = bridge.cv2_to_imgmsg(undistorted_frame, encoding="bgr8")
    undistorted_image_msg.header.stamp = rospy.get_rostime()
    undistorted_image_pub.publish(undistorted_image_msg)

    # Optionally display the images
    # cv2.imshow('Distorted Frame', frame)
    # cv2.imshow('Undistorted Frame', undistorted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
