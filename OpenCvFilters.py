import cv2
import numpy as np


def open_img():
    img_path = "1.jpg"
    load_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return load_img

def choose_filter():
    filters = []
    edge_kernel = np.array(([-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]), dtype=int)
    identity_kernel = np.array(([0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]), dtype=int)
    sharpen_kernel = np.array(([0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]), dtype=int)

    boxBlur_kernel = np.array(([1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]), dtype=int)
    gaussianBlur_kernel = np.array(([1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]), dtype=int)

    filters.append(edge_kernel)
    filters.append(identity_kernel)
    filters.append(sharpen_kernel)
    filters.append(boxBlur_kernel)
    filters.append(gaussianBlur_kernel)

    return filters

def covolution(filter,img):

    #edge detection matrix RGB
    choice = int(input("give choise for filter:"))

    img_edge = cv2.filter2D(img, -1, filter[choice])

    if(choice == 1):
        new = cv2.cvtColor(img_edge, cv2.COLOR_BGR2GRAY)
        cv2.imshow("test", new)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        cv2.imshow("test", img_edge)
        cv2.waitKey()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    filter = []
    load_img = open_img()
    filter = choose_filter()
    covolution(filter, load_img)
