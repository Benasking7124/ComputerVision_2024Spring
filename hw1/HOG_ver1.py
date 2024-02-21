import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
Do not change the input/output of each function, and do not remove the provided functions.
'''

def get_differential_filter():
    # To do
    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return filter_x, filter_y


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


def get_gradient(im_dx, im_dy):
    # To do
    (rows, columns) = im_dx.shape
    grad_mag = np.empty([rows, columns])
    grad_angle = np.empty([rows, columns])

    for i in range(rows):
        for j in range(columns):
            grad_mag[i][j] = np.sqrt(im_dx[i][j] ** 2 + im_dy[i][j] ** 2)
            grad_angle[i][j] = (np.arctan2(im_dy[i][j], im_dx[i][j]) + np.pi) % np.pi
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    (rows, columns) = grad_mag.shape
    M = int(rows / cell_size)
    N = int(columns / cell_size)
    ori_histo = np.zeros([M, N, 6])
    for i in range(M):
        for j in range(N):
            for k in range(cell_size):
                for l in range(cell_size):
                    angle = grad_angle[k + cell_size * i][l + cell_size * j]
                    mag = grad_mag[k + cell_size * i][l + cell_size * j]
                    if ((angle >= 2.8798) and (angle < np.pi)) or ((angle >= 0) and (angle < 0.2618)):
                        ori_histo[i][j][0] += mag
                    elif (angle >= 0.2618) and (angle < 0.7854):
                        ori_histo[i][j][1] += mag
                    elif (angle >= 0.7854) and (angle < 1.3090):
                        ori_histo[i][j][2] += mag
                    elif (angle >= 1.309) and (angle < 1.8326):
                        ori_histo[i][j][3] += mag
                    elif (angle >= 1.8326) and (angle < 2.3562):
                        ori_histo[i][j][4] += mag
                    else:
                        ori_histo[i][j][5] += mag
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    ori_histo_normalized = np.empty([(ori_histo.shape[0] - (block_size - 1)), (ori_histo.shape[1] - (block_size - 1)), (6 * block_size * block_size)])
    for i in range(ori_histo_normalized.shape[0]):
        for j in range(ori_histo_normalized.shape[1]):
            h_sum = 0
            for k in range(6):
                for l in range(block_size):
                    for m in range(block_size):
                        ori_histo_normalized[i][j][k + (l + m * block_size) * 6] = ori_histo[i + l][j + m][k]
                        h_sum += ori_histo[i + l][j + m][k] ** 2

            h_sum = np.sqrt(h_sum + 0.001 ** 2)
            for k in range(ori_histo_normalized.shape[2]):
                ori_histo_normalized[i][j][k] /= h_sum

    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do
    fx, fy = get_differential_filter()
    ix = filter_image(im, fx)
    iy = filter_image(im, fy)
    grad_mag, grad_ang = get_gradient(ix, iy)
    histo = build_histogram(grad_mag, grad_ang, 8)
    hog = get_block_descriptor(histo, 2)

    # visualize to verify
    # visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def face_recognition(I_target, I_template):
    # To do
    bounding_boxes = np.empty([0, 3])

    # HoG over whole target version
    # hog_template = extract_hog(I_template)
    # hog_mean = np.mean(hog_template)
    # for i in range(hog_template.shape[0]):
    #     for j in range(hog_template.shape[1]):
    #         for k in range(hog_template.shape[2]):
    #             hog_template[i][j][k] -= hog_mean

    # hog_target = extract_hog(I_target)
    # hog_mean = np.mean(hog_target)
    # for i in range(hog_target.shape[0]):
    #     for j in range(hog_target.shape[1]):
    #         for k in range(hog_target.shape[2]):
    #             hog_target[i][j][k] -= hog_mean

    # for i in range(hog_target.shape[0] - hog_template.shape[0]):
    #     for j in range(hog_target.shape[1] - hog_template.shape[1]):
    #         s_i = 0
    #         norm_template = 0
    #         norm_target = 0
    #         for k in range(hog_template.shape[0]):
    #             for l in range(hog_template.shape[1]):
    #                 for m in range(24):
    #                     norm_template += hog_template[k][l][m] ** 2
    #                     norm_target += hog_target[i + k][j + l][m] ** 2
    #                     s_i += hog_template[k][l][m] * hog_target[i + k][j + l][m]
    #         norm_template = np.sqrt(norm_template)
    #         norm_target = np.sqrt(norm_target)
    #         s_i = s_i / (norm_template * norm_target)
    #         if (s_i > 0.6):
    #             bounding_boxes = np.append(bounding_boxes, np.array([[j * 8, i * 8, s_i]]), axis=0)

    # HoG individully version
    hog_template = extract_hog(I_template).flatten()
    hog_mean = np.mean(hog_template)
    for i in range(hog_template.shape[0]):
        hog_template[i] -= hog_mean

    iterate_step = 4
    for i in range(0, I_target.shape[0] - I_template.shape[0], iterate_step):
        print(i)
        for j in range(0, I_target.shape[1] - I_template.shape[1], iterate_step):
            I_target_sub = np.empty(I_template.shape)
            for k in range(I_template.shape[0]):
                for l in range(I_template.shape[1]):
                    I_target_sub[k][l] = I_target[i + k][j + l]
            hog_target = extract_hog(I_target_sub).flatten()
            hog_mean = np.mean(hog_target)
            for k in range(hog_target.shape[0]):
                hog_target[k] -= hog_mean

            s_i = 0
            norm_template = 0
            norm_target = 0
            for k in range(hog_template.shape[0]):
                norm_template += hog_template[k] ** 2
                norm_target += hog_target[k] ** 2
                s_i += hog_template[k] * hog_target[k]
            norm_template = np.sqrt(norm_template)
            norm_target = np.sqrt(norm_target)
            s_i = s_i / (norm_template * norm_target)

            if (s_i > 0.43):
                bounding_boxes = np.append(bounding_boxes, np.array([[j, i, s_i]]), axis=0)
    
    # Non Maximum Suppression
    two_bounding_boxes_area = I_template.shape[0] * I_template.shape[0] * 2
    bounding_boxes = np.array(sorted(bounding_boxes, reverse=True, key = lambda x:x[2]))
    for i in range(bounding_boxes.shape[0]):
        # print(bounding_boxes)
        delete_list = []

        # Calculate IoU
        for j in range(i + 1, bounding_boxes.shape[0]):
            IoU = 0
            intersection = 0
            for x in range(int(bounding_boxes[i][0]), (int(bounding_boxes[i][0]) + I_template.shape[0])):
                for y in range(int(bounding_boxes[i][1]), (int(bounding_boxes[i][1]) + I_template.shape[0])):
                    if (x >= bounding_boxes[j][0]) and (x <= (bounding_boxes[j][0] + I_template.shape[0])) and (y >= bounding_boxes[j][1]) and (y <= (bounding_boxes[j][1] + I_template.shape[0])):
                        intersection += 1
            IoU = intersection / (two_bounding_boxes_area - intersection)

            if (IoU > 0.5):
                delete_list.append(j)
        bounding_boxes = np.delete(bounding_boxes, delete_list, 0)
    
    print(bounding_boxes)

    return bounding_boxes


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()




if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)

    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    #this is visualization code.
