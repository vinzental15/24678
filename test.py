import cv2
import numpy as np

def depth_map(disp, frame_name, img_l):
    h, w = img_l.shape[:2]
    f = 0.8 * w  # guess the focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],
                    [0, 0, 0, -f],  # This should be the focal length times the baseline (negative if depth is positive)
                    [0, 0, 1, 0]])  # Disparity-to-depth mapping
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_points = out_points.reshape(-1, 3)
    out_colors = out_colors.reshape(-1, 3)

    # Create a point cloud data
    verts = np.hstack([out_points, out_colors])
    with open(frame_name, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d')

# StereoSGBM Parameter Definitions
min_disp = 0
num_disp = 160
window_size = 5
block_size = window_size * 2 + 1
P1 = 8 * 3 * block_size**2
P2 = 32 * 3 * block_size**2

# Create StereoSGBM and compute disparity
left_matcher = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=P1,
    P2=P2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.StereoSGBM_MODE_SGBM_3WAY
)

# Read left and right images and convert to grayscale
img_l_color = cv2.imread("ball-left.png")  # Load in color for color point cloud
img_r = cv2.imread("ball-right.png", cv2.IMREAD_GRAYSCALE)
img_l = cv2.cvtColor(img_l_color, cv2.COLOR_BGR2GRAY)

# Compute the disparity map
disparity = left_matcher.compute(img_l, img_r).astype(np.float32) / 16.0

# Normalize the disparity map for visualization
disp_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow("Disparity", disp_visual)  # Show the normalized disparity map

# Generate 3D point cloud
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
frame_name = 'ball.ply'
depth_map(disparity, frame_name, img_l_color)

cv2.waitKey(0)



