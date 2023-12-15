import cv2
import numpy as np
import glob

# Define the chess board rows and columns
rows = 8
cols = 5

# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((rows*cols, 3), np.float32)
objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

# Arrays to store object points and image points from all images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# List of calibration images
images = glob.glob('/home/osama/Desktop/elp/calibration_elp/*.png')

image_shape = None  # Initialize image shape

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load image {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_shape is None:
        image_shape = gray.shape[::-1]

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if not ret:
        print(f"Chessboard corners not found in image {fname}")
        continue

    # Each entry in objpoints should be a copy of objp
    objpoints.append(objp.copy())  # Use copy of objp for each successful detection
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    cv2.drawChessboardCorners(img, (cols, rows), corners2, ret)
    cv2.imshow('img', img)
    cv2.waitKey(500)

cv2.destroyAllWindows()

# Check if object points and image points were found
if objpoints and imgpoints and image_shape:
    # Convert lists to NumPy arrays with the correct dtype
    objpoints_np = np.array(objpoints, dtype=np.float32)
    imgpoints_np = np.array(imgpoints, dtype=np.float32)
    # objpoints_np.reshape(-1,1,3)    # np.expand_dims(np.asarray(imgpoints_np), -2)
    objpoints_np = np.expand_dims(np.asarray(objpoints_np), -2)
    # import pdb
    # pdb.set_trace()
    # Calibrate fisheye camera model
    N_OK = len(objpoints_np)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints_np,
        imgpoints_np,
        image_shape,
        K,
        D,
        rvecs,
        tvecs,
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW,
        criteria
    )

    print("Found " + str(N_OK) + " valid images for calibration")
    print("Camera matrix:", K)
    print("Distortion coefficients:", D)
else:
    print("Insufficient valid images for calibration.")
