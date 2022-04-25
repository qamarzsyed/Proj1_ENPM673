import numpy as np
import cv2
import scipy

K = np.array([[1346.100595, 0, 932.1633975], [0, 1355.933136, 654.8986796], [0, 0, 1]])

# shifts image into fft space and then creates a mask w radius 250 to filter out low frequency
# then returns to image space with backgroung filtered out
def fft(img):
    ydim, xdim, zdim = img.shape
    x_center, y_center = int(ydim / 2), int(xdim / 2)
    center = [x_center, y_center]
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = cv2.dft(np.float32(grayscale), flags=cv2.DFT_COMPLEX_OUTPUT)
    f = np.fft.fftshift(f)
    highpass = np.ones((ydim, xdim, 2), np.uint8)
    r = 250
    x, y = np.ogrid[:ydim, :xdim]
    low_freq = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    highpass[low_freq] = 0
    f_filt = f * highpass
    f = np.fft.ifftshift(f_filt)
    edges = cv2.idft(f)
    edges = cv2.magnitude(edges[:, :, 0], edges[:, :, 1])
    return edges

    # corners = cv2.goodFeaturesToTrack(edges, 25, 0.01, 50)
    # corners = np.int0(corners)
    # for i in corners:
    #     x, y = i.ravel()
    #     cv2.circle(edges, (x, y), 3, 255, 10)
    # cv2.imshow('dkfaj', edges)
    # cv2.waitKey(0)

# returns coords of smallest rectangle that can surround a group of corners
def rect_coords(img, min_dist, max_can):
    img = cv2.Canny(img, 100, max_can)
    corners = cv2.goodFeaturesToTrack(img, 30, 0.1, min_dist)
    corners = np.int0(corners)
    rect = cv2.minAreaRect(corners)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

# uses Canny edge detector and Shi-tomasi corner detector to find corners
# makes smallest rectangle possible form those corners and returns the rectangle as it would be seen at its angle
def image_crop(img, min_dist, max_can):
    init = img
    img = cv2.Canny(img, 100, max_can)
    corners = cv2.goodFeaturesToTrack(img, 30, 0.1, min_dist)
    corners = np.int0(corners)

    for num in corners:
        x, y = num.ravel()
        cv2.circle(img, (x, y), 3, 255, 10)
    rect = cv2.minAreaRect(corners)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img_rot = cv2.warpAffine(init, M, (cols, rows))
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0
    img_crop = img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]
    return img_crop

# edge detects and crops borders twice until it gets an image that is just the tag
# filters and thresholds the image until it is a binary grid of black and white
def binary_ar(img):
    paper = image_crop(img, 50, 800)
    x = 20
    paper = paper[x:-x, x:-x]
    l_tag = image_crop(paper, 50, 800)
    l_tag = l_tag[x:-x, x:-x]
    l_tag = cv2.cvtColor(l_tag, cv2.COLOR_BGR2GRAY)
    ar_tag = cv2.Canny(l_tag, 100, 200)
    ar_tag = image_crop(l_tag, 10, 200)
    ar_tag = cv2.resize(ar_tag, (4, 4), interpolation=cv2.INTER_LINEAR)
    a, ar_tag_binary = cv2.threshold(ar_tag, 200, 255, cv2.THRESH_BINARY)
    ar_array = ar_tag_binary/255
    return ar_array

# prints out array value and converted code
def binary_code(bin_array):
    rotations = 0
    if bin_array[0, 0] == 1:
        rotations = 2
    elif bin_array[3, 0] == 1:
        rotations = 1
    elif bin_array[0, 3] == 1:
        rotations = 3
    bin_array = np.rot90(bin_array, rotations)
    print(bin_array)
    tag_id = bin_array[1, 1] * 2 ** 0 + bin_array[1, 2] * 2 ** 1 + bin_array[2, 2] * 2 ** 2 + bin_array[2, 1] * 2 ** 3
    print(tag_id)

# uses smallest eigenvalue to make Homography matrix for image and ar tag borders
def homography(tag, img):
    x_img, y_img, z_img = img.shape
    x_1, x_2, x_3, x_4 = x_img, 0, 0, x_img
    y_1, y_2, y_3, y_4 = y_img, y_img, 0, 0
    paper_coords = rect_coords(tag, 50, 800)
    x_c = np.mean(paper_coords[:,0])
    y_c = np.mean(paper_coords[:,1])
    mean = [x_c, y_c]
    paper_coords = mean - 0.40*(mean-paper_coords)
    # x_min, y_min = paper_coords.min(axis=0)
    # addon = (x_min, y_min)
    # x = 20
    # paper = paper[x:-x, x:-x]
    # tag_coords = rect_coords(paper, 50, 800)
    # tag_coords = tag_coords + addon
    x1, x2, x3, x4 = paper_coords[0, 0], paper_coords[1, 0], paper_coords[2, 0], paper_coords[3, 0]
    y1, y2, y3, y4 = paper_coords[0, 1], paper_coords[1, 1], paper_coords[2, 1], paper_coords[3, 1]

    A = np.array([[-x1, -y1, -1, 0, 0, 0, x1 * x_1, y1 * x_1, x_1],
                       [0, 0, 0, -x1, -y1, -1, x1 * y_1, y1 * y_1, y_1],
                       [-x2, -y2, -1, 0, 0, 0, x2 * x_2, y2 * x_2, x_2],
                       [0, 0, 0, -x2, -y2, -1, x2 * y_2, y2 * y_2, y_2],
                       [-x3, -y3, -1, 0, 0, 0, x3 * x_3, y3 * x_3, x_3],
                       [0, 0, 0, -x3, -y3, -1, x3 * y_3, y3 * y_3, y_3],
                       [-x4, -y4, -1, 0, 0, 0, x4 * x_4, y4 * x_3, x_4],
                       [0, 0, 0, -x4, -y4, -1, x4 * y_4, y4 * y_3, y_4]])

    e_vals, e_vecs = np.linalg.eig(np.dot(A.T, A))
    H = e_vecs[:, np.argmin(e_vals)]
    H = np.reshape(H, (3, 3))
    return H

# puts turtle on top of AR tag
def apply_homography(img, H, outer):
    h, w, d = outer.shape
    ind_y, ind_x = np.indices((h, w), dtype=np.float32)
    index_linearized = np.array([ind_x.ravel(), ind_y.ravel(), np.ones_like(ind_x).ravel()])

    map_ind = H.dot(index_linearized)
    map_x, map_y = map_ind[:-1] / map_ind[-1]
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)
    warped_img = np.zeros((h, w, 3), dtype="uint8")
    map_x[map_x >= img.shape[1]] = -1
    map_x[map_x < 0] = -1
    map_y[map_y >= img.shape[0]] = -1
    map_y[map_y < 0] = -1

    for new_x in range(w):
        for new_y in range(h):
            x = int(map_x[new_y, new_x])
            y = int(map_y[new_y, new_x])

            if x == -1 or y == -1:
                pass
            else:
                warped_img[new_y, new_x] = img[y, x]
    warped_img[warped_img == 0] = outer[warped_img == 0]
    return warped_img

# makes homography matrix, then P matrix, then inverses P and applies to points to get homogeneous coords
# then adds height to those coords, then applies regular P to go back to cartesian and plots dots
def draw_cube(tag):
    paper_coords = rect_coords(tag, 50, 800)
    x_c = np.mean(paper_coords[:, 0])
    y_c = np.mean(paper_coords[:, 1])
    mean = [x_c, y_c]
    paper_coords = mean + 0.40 * (mean - paper_coords)
    x1, x2, x3, x4 = paper_coords[0, 0], paper_coords[1, 0], paper_coords[2, 0], paper_coords[3, 0]
    y1, y2, y3, y4 = paper_coords[0, 1], paper_coords[1, 1], paper_coords[2, 1], paper_coords[3, 1]

    z = 20
    ar_tag = image_crop(cv2.cvtColor(image_crop(tag, 50, 800)[z:-z, z:-z], cv2.COLOR_BGR2GRAY), 10, 200)
    square_dim, _ = ar_tag.shape
    square_dim = 1

    x_img = square_dim
    y_img = square_dim
    z_img = square_dim
    x_1, x_2, x_3, x_4 = x_img, 0, 0, x_img
    y_1, y_2, y_3, y_4 = y_img, y_img, 0, 0


    A = np.array([[-x1, -y1, -1, 0, 0, 0, x1 * x_1, y1 * x_1, x_1],
                  [0, 0, 0, -x1, -y1, -1, x1 * y_1, y1 * y_1, y_1],
                  [-x2, -y2, -1, 0, 0, 0, x2 * x_2, y2 * x_2, x_2],
                  [0, 0, 0, -x2, -y2, -1, x2 * y_2, y2 * y_2, y_2],
                  [-x3, -y3, -1, 0, 0, 0, x3 * x_3, y3 * x_3, x_3],
                  [0, 0, 0, -x3, -y3, -1, x3 * y_3, y3 * y_3, y_3],
                  [-x4, -y4, -1, 0, 0, 0, x4 * x_4, y4 * x_3, x_4],
                  [0, 0, 0, -x4, -y4, -1, x4 * y_4, y4 * y_3, y_4]])

    e_vals, e_vecs = np.linalg.eig(np.dot(A.T, A))
    H = e_vecs[:, np.argmin(e_vals)]
    H = np.reshape(H, (3, 3))
    H = H * (-1)

    B_H = np.dot(np.linalg.inv(K), H)
    if np.linalg.norm(B_H) > 0:
        B = 1 * B_H
    else:
        B = -1 * B_H
    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]
    lambda_ = np.sqrt(np.linalg.norm(b1, 2) * np.linalg.norm(b2, 2))
    r1 = b1 / lambda_
    r2 = b2 / lambda_
    trans = b3 / lambda_
    c = r1 + r2
    p = np.cross(r1, r2)
    d = np.cross(c, p)
    r1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / np.sqrt(2))
    r2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / np.sqrt(2))
    r3 = np.cross(r1, r2)
    Rt = np.stack((r1, r2, r3, trans)).T
    P = np.dot(K, Rt)

    x = []
    y = []
    z = []

    for point in paper_coords:
        x.append(point[0])
        y.append(point[1])
        z.append(square_dim)

    Pinv = np.linalg.pinv(P)
    print(P.shape)
    p1 = np.array([x[0], y[0], 1])
    p2 = np.array([x[1], y[1], 1])
    p3 = np.array([x[2], y[2], 1])
    p4 = np.array([x[3], y[3], 1])

    p1 = np.dot(Pinv, p1)
    p1[2] = square_dim
    p2 = np.dot(Pinv, p2)
    p2[2] = square_dim
    p3 = np.dot(Pinv, p3)
    p3[2] = square_dim
    p4 = np.dot(Pinv, p4)
    p4[2] = square_dim

    p1 = np.dot(P, p1)
    p2 = np.dot(P, p2)
    p3 = np.dot(P, p3)
    p4 = np.dot(P, p4)

    p_list = [p1,p2,p3,p4]
    p_list = np.array(p_list)

    corners = paper_coords
    corners = corners.astype(int)
    for corner in corners:
        x, y = corner
        cv2.circle(tag, (x, y) , 3, 255, 10)

    p_list_int = p_list.astype(int)
    for p in p_list_int:
        x, y, z = p
        cv2.circle(tag, (int(x/z), int(y/z)), 3, 255, 10)

    return tag



