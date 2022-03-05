import cv2
import numpy as np

convol_list = []

def open_img():
    img_path = "enemy.jpg"
    load_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return load_img

def choose_filter():
    filters = []

    ridge_kernel = np.array(([-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]), dtype=np.int32)

    identity_kernel = np.array(([0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]), dtype=np.int32)

    sharpen_kernel = np.array(([0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]), dtype=np.int32)

    boxBlur_kernel = np.array(([1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]), dtype=np.int32)

    gaussianBlur_kernel = np.array(([1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]), dtype=np.int32)


    filters.append(ridge_kernel)
    filters.append(identity_kernel)
    filters.append(sharpen_kernel)
    filters.append(boxBlur_kernel)
    filters.append(gaussianBlur_kernel)

    return filters

# pick a kernel, convolute the img and save it as test.jpg
def convolution(filter,img):
    # final 1237, 1856
    c_filter = filter[0]

    k_size = len(c_filter)

    i_padded = np.pad(img, (k_size -3, k_size -3))

    image_w, image_h = img.shape
    
    for row in range(image_w - 3):
        for col in range(image_h - 3):
            convol_list.append(np.sum(i_padded[row:k_size+row, col:k_size+col] * c_filter))
    
    convol_arr = np.array(convol_list, dtype=np.float32)
    convol_arr.shape = (-1, int(image_w - 3))
    
    cv2.imwrite('test.jpg', convol_arr)

if __name__ == "__main__":
    filter = []
    load_img = open_img()
    filter = choose_filter()
    convolution(filter, load_img)
