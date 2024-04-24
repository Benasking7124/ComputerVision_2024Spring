import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D


def compute_F(pts1, pts2):
    # TO DO
    F = np.empty([3, 3])
    max_n_inlier = 0
    max_inliers = []

    for _ in range(100):
        n_inlier = 0
        inliers = []
        n = np.random.choice(range(pts1.shape[0]), size = 8, replace=False)

        # Construct A
        A = np.empty([0, 9], dtype=np.float32)
        for i in range(8):
            x1, y1 = pts1[n[i]]
            x2, y2 = pts2[n[i]]
            # A = np.vstack([A, [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]])
            A = np.vstack([A, [x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1]])
        
        # Compute SVD of A
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        F_temp = vh[-1].reshape(3, 3)

        # SVD Cleanup
        u, s, vh = np.linalg.svd(F_temp, full_matrices=False)
        s[-1] = 0
        F_temp = u @ np.diag(s) @ vh

        # Count inlier
        for i in range(pts1.shape[0]):
            p1 = np.hstack([pts1[i], 1])[:, None]
            p2 = np.hstack([pts2[i], 1])[None, :]
            if (p2 @ F_temp @ p1) < 1e-6:
                inliers.append(i)
                n_inlier +=1

        if (n_inlier > max_n_inlier):
            max_n_inlier = n_inlier
            max_inliers = list(inliers)

    # Recalculate F based on inliers
    p1 = np.take(pts1, max_inliers, axis=0)
    p2 = np.take(pts2, max_inliers, axis=0)
    A = np.empty([0, 9], dtype=np.float32)
    for i in range(p1.shape[0]):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A = np.vstack([A, [x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1]])
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    F = vh[-1].reshape(3, 3)
    u, s, vh = np.linalg.svd(F, full_matrices=False)
    s[-1] = 0
    F = u @ np.diag(s) @ vh
    
    return F


def triangulation(P1, P2, pts1, pts2):
    # TO DO
    pts3D = np.empty([0, 3])
    
    for i in range(pts1.shape[0]):
        pts1_skew = np.array([[0, -1, pts1[i][1]],
                              [1, 0, -pts1[i][0]],
                              [-pts1[i][1], pts1[i][0], 0]])
        A_up = pts1_skew @ P1
        pts2_skew = np.array([[0, -1, pts2[i][1]],
                              [1, 0, -pts2[i][0]],
                              [-pts2[i][1], pts2[i][0], 0]])
        A_down = pts2_skew @ P2

        A = np.vstack([A_up, A_down])
        _, _, vh = np.linalg.svd(A)
        pt_3D = vh[-1][:3] / vh[-1][3]
        pts3D = np.vstack([pts3D, pt_3D])
    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    # TO DO
    n_valid = [0, 0, 0, 0]

    for i in range(len(Rs)):
        # Camera 1
        T1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        T1 = np.vstack([T1, [0, 0, 0, 1]])

        # Camera 2
        T2 = np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        T2 = np.vstack([T2, [0, 0, 0, 1]])

        # Check Z coordinate
        for j in range(pts3Ds[i].shape[0]): 
            p_c1 = T1 @ np.hstack([pts3Ds[i][j], 1])[:, None]
            p_c2 = T2 @ np.hstack([pts3Ds[i][j], 1])[:, None]
            if (p_c1[2] > 0) and (p_c2[2] > 0):
                n_valid[i] += 1

    max_valid = n_valid.index(max(n_valid))
    R = Rs[max_valid]
    C = Cs[max_valid]
    pts3D = pts3Ds[max_valid]
    return R, C, pts3D


def compute_rectification(K, R, C):
    # TO DO
    r1 = (C / np.linalg.norm(C)).ravel()
    r2 = (np.array([-C[1][0], C[0][0], 0]) / np.sqrt(C[0][0] ** 2 + C[1][0] ** 2))
    r3 = np.cross(r1, r2)
    R_rect = np.vstack([r1.T, r2.T, r3.T])

    H1 = K @ R_rect @ np.linalg.inv(K)
    H2 = K @ R_rect @ R.T @ np.linalg.inv(K)
    return H1, H2


def dense_match(img1, img2, descriptors1, descriptors2):
    # TO DO
    disparity = np.empty(img1.shape)

    for i in range(img1.shape[0]):
        print(i)
        for j in range(img1.shape[1]):
            d = descriptors1[i][j]
            knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(descriptors2[i][j:img1.shape[1]])
            _, n = knn.kneighbors(descriptors1[i][j][None, :])
            disparity[i][j] = -n[0][0]
            # for k in range(j, img1.shape[1]):
            #     dk = descriptors2[i][k]
            #     difference
    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = (F @ np.array([[p[0], p[1], 1]]).T).flatten()
    p1, p2 = (0, int(-el[2] / el[1])), (img.shape[1], int((-img_width * el[0] - el[2]) / el[1]))
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
        ax.title.set_text('Configuration {}'.format(i))
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
        ax.title.set_text('Configuration {}'.format(i))
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)
    C = C.flatten()  # (3, 1) -> (3,)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    disparity[disparity > 150] = 150
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 0: get correspondences between image pair
    data = np.load('./correspondence.npz')
    pts1, pts2 = data['pts1'], data['pts2']
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 1: compute fundamental matrix and recover four sets of camera poses
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 2: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 3: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 4: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 5: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    data = np.load('./dsift_descriptor.npz')
    desp1, desp2 = data['descriptors1'], data['descriptors2']
    disparity = dense_match(img_left_w, img_right_w, desp1, desp2)
    visualize_disparity_map(disparity)
