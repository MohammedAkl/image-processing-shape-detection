import cv2
import numpy as np
import argparse


def read_img():
    ap = argparse.ArgumentParser()
    # Add your images path
    ap.add_argument("-input", type=str,
                    default="C:/Users/Blu-Ray/Desktop/image-stitching-opencv/images/output/finalgrid.png",
                    help="path to input directory of images to stitch")
    args = vars(ap.parse_args())
    frame = cv2.imread(args["input"])
    return frame


def shape_color(frame):
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    # COLORS
    # BLUE
    blue_lower = np.array([110, 50, 50])
    blue_upper = np.array([130, 255, 255])
    # ADD MASKS
    mask = cv2.inRange(hsv, blue_lower, blue_upper)
    return mask


def shape_contour(mask, frame):
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    x3 = 0
    y3 = 0
    x4 = 0
    y4 = 0
    cent, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in cent:
        area = cv2.contourArea(contour)
        approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
        if area > 500:
            cv2.drawContours(frame, [approx], -2, (255, 255, 255), 6)
            n = approx.ravel()
            i = 0
            for j in n:
                if i % 2 == 0:
                    x = n[i]
                    y = n[i + 1]
                if i == 0:
                    if x1 == 0:
                        x1 = x
                        y1 = y
                    else:
                        x2 = x
                        y2 = y
                elif i == 2:
                    if x3 == 0:
                        x3 = x
                        y3 = y
                    else:
                        x4 = x
                        y4 = y
                i = i + 1
    return x1, x4, y3, y4


def shape_show(mask, frame, x1, x4, y3, y4):
    # res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("frame", frame)
    # cv2.imshow("mask", mask)
    # cv2.imshow("res", res)
    cropped_image = frame[y4:y3, x1:x4]
    cv2.imshow("cropped", cropped_image)
    return cropped_image


def cropped_dim(cropped_image):
    # get dimension of cropped image for only once`
    dimensions = cropped_image.shape
    # height and width in image
    height = cropped_image.shape[0]
    width = cropped_image.shape[1]
    print('Image Dimension    : ', dimensions)
    print('Image Height       : ', height)
    print('Image Width        : ', width)
    return height, width


def make_grid(r, c):
    grid = np.ones((r, c, 3)) * 255

    for i in range(0, r, r // 3):
        cv2.line(grid, (0, i), (c, i), (0, 0, 0), 2)

    for j in range(0, c, c // 9):
        cv2.line(grid, (j, 0), (j, r), (0, 0, 0), 2)

    return grid


def mask_cropped(cropped_image, height, width):
    # calculate every square in grid
    h = int(height / 3)
    w = int(width / 9)
    blurred_frame = cv2.GaussianBlur(cropped_image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    # COLORS
    # BLACK = 1
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([255, 255, 80])
    # YELLOW = 2
    yellow_lower = np.array([0, 245, 0])
    yellow_upper = np.array([255, 255, 255])
    # RED = 3
    red_lower = np.array([100, 100, 0])
    red_upper = np.array([255, 255, 255])

    # ADD MASKS
    l = []
    mask_black = cv2.inRange(hsv, black_lower, black_upper)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_red = cv2.inRange(hsv, red_lower, red_upper)
    color_mask(mask_black, cropped_image, l, h, w, 1, w, h)
    color_mask(mask_yellow, cropped_image, l, h, w, 2, w, h)
    color_mask(mask_red, cropped_image, l, h, w, 3, w, h)
    print(l)
    cv2.imshow(" mask cropped ", cropped_image)
    # cv2.imshow("yellow",mask_yellow)

    return l


def color_mask(mask, cropped_image, l, h, w, color, w1, h1):
    font = cv2.FONT_HERSHEY_COMPLEX
    cent, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    l1=[]
    #print(w1,h1)
    for c in cent:
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
        # compute the center of the contour
        if area > 100:
            M = cv2.moments(c)
            n = approx.ravel()
            i = 0
            xT,yT,wT,hT=cv2.boundingRect(c)
            #print(xT,yT,wT,hT)
            if(wT>w):
                print(xT, yT, wT, hT)
                for j in n:
                    if (i % 2 == 0):
                        x = n[i]
                        y = n[i + 1]
                    i = i + 1
                cXT1 = xT
                cYT1 = yT
                cXT2 = xT+wT
                cYT2 = yT
                # print(cX, cY)
                cv2.circle(cropped_image, (cXT1, cYT1), 3, (0, 100, 255), -1)
                xT1 = int(cXT1 / w)
                yT1 = int(cYT1 / h)
                cor = [xT1, yT1, 4]
                l.append(cor)
                cv2.circle(cropped_image, (cXT2, cYT2), 3, (0, 100, 255), -1)
                xT2 = int(cXT2 / w)
                yT2 = int(cYT2 / h)
                cor = [xT2, yT2, 4]
                l.append(cor)
            else:
                for j in n:
                    if (i % 2 == 0):
                        x = n[i]
                        y = n[i + 1]
                    i = i + 1
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # print(cX, cY)
                cv2.circle(cropped_image, (cX, cY), 3, (0, 100, 255), -1)
                x1 = int(cX / w)
                y1 = int(cY / h)
                cor = [x1, y1, color]
                l.append(cor)
                #print(area)
            cv2.drawContours(cropped_image, [approx], -1, (0, 255, 0), 2)
    return l


def grid_dim(grid):
    dimensions = grid.shape
    # height and width in image
    height = grid.shape[0]
    width = grid.shape[1]
    '''
    print('Grid Dimension    : ', dimensions)
    print('Grid Height       : ', height)
    print('Grid Width        : ', width)
    '''
    return height, width


def grid_in_draw(grid, l, height, width):
    h = int(height / 3)
    w = int(width / 9)
    radius = 30
    thickness = 2
    for i in l:
        x, y = i[:2]
        c = i[2]
        center_coordinates = (int((x * w) + (w / 2)), int((y * h) + (h / 2)))
        #print(c)
        if c == 1:
            color = (0, 128, 0)
        elif c == 2:
            color = (0,255,255)
        elif c == 3:
            color = (255, 0, 0)
        elif c == 4:
            color = (0, 0, 255)



        cv2.circle(grid, center_coordinates, radius, color, thickness)
    return grid


def show_grid(grid):
    cv2.imshow("Grid", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


# MAIN
f = read_img()
m = shape_color(f)
x1, x2, y1, y2 = shape_contour(m, f)
cropped = shape_show(m, f, x1, x2, y1, y2)
h, w = cropped_dim(cropped)
l = mask_cropped(cropped, h, w)
g = make_grid(300, 900)
hg, wg = grid_dim(g)
newG = grid_in_draw(g, l, hg, wg)
show_grid(newG)
