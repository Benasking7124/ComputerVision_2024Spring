import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate


def find_match(img1, img2):
    # To do
    sift = cv2.SIFT_create()
    kp1, descriptor1 = sift.detectAndCompute(img1, None)
    kp2, descriptor2 = sift.detectAndCompute(img2, None)

    # matching img2 to img1
    knn2 = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(descriptor2)
    distance12, neighbors12 = knn2.kneighbors(descriptor1)
    keep_list = []
    for i in range(len(distance12)):
        if (distance12[i][0] / distance12[i][1]) < 0.7:
            keep_list.append(i)

    matching12 = []
    for x in keep_list:
        matching12.append((kp1[x].pt, kp2[neighbors12[x][0]].pt))

    # matching img1 to img2
    knn1 = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(descriptor1)
    distance21, neighbors21 = knn1.kneighbors(descriptor2)
    keep_list = []
    for i in range(len(distance21)):
        if (distance21[i][0] / distance21[i][1]) < 0.7:
            keep_list.append(i)
    
    matching21 = []
    for x in keep_list:
        matching21.append((kp1[neighbors21[x][0]].pt, kp2[x].pt))

    # bi-directional consistency check
    x1 = np.empty([0, 2])
    x2 = np.empty([0, 2])
    for p1 in matching12:
        for p2 in matching21:
            if p1 == p2:
                x1 = np.append(x1, np.array([p1[0]]), axis=0)
                x2 = np.append(x2, np.array([p1[1]]), axis=0)

    return x1, x2

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    A = np.empty([4, 4])
    max_n_inlier = 0
    max_inliers = []

    for i in range(ransac_iter):
        n_inlier = 0
        inliers = []
        # Random sampling: randomly select 3 points
        n = np.random.default_rng().choice(range(x1.shape[0]), size = 3, replace=False)
        
        # Model building: compute homography
        point1_array = np.array([[x1[n[0]][0], x1[n[0]][1], 1, 0, 0, 0],
                                 [0, 0, 0, x1[n[0]][0], x1[n[0]][1], 1],
                                 [x1[n[1]][0], x1[n[1]][1], 1, 0, 0, 0],
                                 [0, 0, 0, x1[n[1]][0], x1[n[1]][1], 1],
                                 [x1[n[2]][0], x1[n[2]][1], 1, 0, 0, 0],
                                 [0, 0, 0, x1[n[2]][0], x1[n[2]][1], 1],])
        point2_array = np.transpose(np.array([[x2[n[0]][0], x2[n[0]][1], x2[n[1]][0], x2[n[1]][1], x2[n[2]][0], x2[n[2]][1]]]))
        h_param = np.matmul(np.linalg.pinv(point1_array), point2_array)
        A_temp = np.array([[h_param[0][0], h_param[1][0], h_param[2][0]],
                           [h_param[3][0], h_param[4][0], h_param[5][0]],
                           [0, 0, 1]])

        # Tresholding & Inlier counting: compute inliers
        for i in range(x1.shape[0]):
            x2_est = np.matmul(A_temp, np.transpose([x1[i][0], x1[i][1], 1]))
            x2_est = np.array([(x2_est[0] / x2_est[2]), (x2_est[1] / x2_est[2])])
            difference = np.linalg.norm(x2[i] - x2_est)
            if difference < ransac_thr:
                inliers.append(i)
                n_inlier += 1
        
        if (n_inlier > max_n_inlier):
            A = A_temp
            max_n_inlier = n_inlier
            max_inliers = list(inliers)

    # Re-compute H using least-squares
    point1_array = []
    point2_array = []
    for i in max_inliers:
        point1_array.append([x1[i][0], x1[i][1], 1, 0, 0, 0])
        point1_array.append([0, 0, 0, x1[i][0], x1[i][1], 1])
        point2_array.append(x2[i][0])
        point2_array.append(x2[i][1])
    point1_array = np.array(point1_array)
    point2_array = np.array(point2_array)
    h_param= np.linalg.lstsq(point1_array, point2_array)
    A = np.array([[h_param[0][0], h_param[0][1], h_param[0][2]],
                  [h_param[0][3], h_param[0][4], h_param[0][5]],
                  [0, 0, 1]])
    print(max_n_inlier)
    return A

def warp_image(img, A, output_size):
    # To do
    # Get warp coordinate
    x_cor, y_cor = np.meshgrid(range(output_size[1]), range(output_size[0]), indexing = 'xy')
    warp_cooradinate = np.stack((x_cor.reshape(-1), y_cor.reshape(-1), np.ones(output_size).reshape(-1)))
    warp_cooradinate = A @ warp_cooradinate
    warp_cooradinate = warp_cooradinate[:2, :].T

    # Get image coordinate
    img_coordinate = (range(img.shape[1]), range(img.shape[0]))
    img_warped = interpolate.interpn(img_coordinate, img.T, warp_cooradinate, method = 'linear', bounds_error = False, fill_value = None)
    img_warped = img_warped.reshape(output_size)

    return img_warped


def filter_image(im, filter):
    # To do
    (rows, columns) = im.shape
    im_filtered = np.zeros([rows, columns])

    # Zero padding
    padding_nx = int(filter.shape[1] / 2)
    padding_ny = int(filter.shape[0] / 2)
    im_zp = np.pad(im, ((padding_ny, padding_ny), (padding_nx, padding_nx)), 'constant', constant_values=0)
    
    # Convolution
    for i in range(rows):
        for j in range(columns):
            for k in range(filter.shape[0]):
                for l in range(filter.shape[1]):
                    im_filtered[i][j] += im_zp[i + k][j + l] * filter[filter.shape[0] - k - 1][filter.shape[1] - l -1]
    return im_filtered


def align_image(template, target, A):
    # To do
    # initialize p=p0 from input A
    p = np.array([A[0][0], A[0][1], A[0][2], A[1][0], A[1][1], A[1][2]])
    A_refined = A

    # compute the gradient of the template image
    ix = filter_image(template, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    iy = filter_image(template, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))

    # compute the steepest decent images
    I_sd = np.empty([template.shape[0], template.shape[1], 6])
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            I_sd[i][j] = np.matmul(np.array([ix[i][j], iy[i][j]]), np.array([[j, i, 1, 0, 0, 0], [0, 0, 0, j, i, 1]]))

    # compute Hessian
    hessian = np.zeros([6, 6])
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            I_sd_x_2d = I_sd[i][j][np.newaxis, :]
            hessian += np.matmul(np.transpose(I_sd_x_2d), I_sd_x_2d)

    iteration = 0
    while True:
        I_tgt_warp = warp_image(target, A_refined, template.shape)
        I_error = I_tgt_warp - template

        # compute F
        F_function = np.zeros([6, 1])
        for i in range(template.shape[0]):
            for j in range(template.shape[1]):
                I_sd_x_2d_transpose = np.transpose(I_sd[i][j][np.newaxis, :])
                F_function += I_sd_x_2d_transpose * I_error[i][j]
        
        # compute delta_p
        delta_p = np.matmul(np.linalg.inv(hessian), F_function).flatten()
        p = p + delta_p
        A_refined = np.vstack((p.reshape(2, 3),  np.array([0, 0, 1])))

        print("Aligning image..., iteration:", iteration, "norm(delta_p)", np.linalg.norm(delta_p))
        iteration += 1
        if np.linalg.norm(delta_p) < 0.1:
            break
    return A_refined


def track_multi_frames(template, img_list):
    # To do
    x1, x2 = find_match(template, target_list[0])
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    A_list = [A]
    for I_tgt in img_list[1:]:
        A = align_image(template, I_tgt, A)
        A_list.append(A)
    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sum(np.square(x2_t[:, :2] - x2), axis=1)
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack(( np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]), np.ones((5, 1)) )) @ A[:2, :].T

    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y')
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./JS_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./JS_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    # visualize_find_match(template, target_list[0], x1, x2)

    ransac_thr = 1
    ransac_iter = 100
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    # visualize_align_image_using_feature(template, target_list[0], x1, x2, A, ransac_thr)

    # img_warped = warp_image(target_list[0], A, template.shape)
    # plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()

    # img_error = template - img_warped
    # plt.imshow(img_error)
    # plt.axis('off')
    # plt.show()

    A_refined = align_image(template, target_list[1], A)
    visualize_align_image(template, target_list[1], A, A_refined)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)


