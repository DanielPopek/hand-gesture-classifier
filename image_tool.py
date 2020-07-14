import cv2
import numpy as np


def naive_threshold(brg_image, k=10):
    img_hls = cv2.cvtColor(brg_image, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(
        img_hls,
        np.array([0, 0.1 * 255, 0.05 * 255]),
        np.array([15, 0.8 * 255, 0.6 * 255])
    )
    blurred = cv2.blur(mask, (k, k))
    th, im = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
    return im


def remove_background(frame, bg_model):
    print('remove background')
    learning_rate = 0
    fgmask = bg_model.apply(frame, learningRate=learning_rate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=2)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def background_threshold(brg_image, bg_model):
    threshold = 60  # binary threshold
    blur_value = 41  # GaussianBlur parameter
    img = remove_background(brg_image, bg_model)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def place_roi_in_whole_img(cv_image, roi):
    if roi.shape == (546, 546):
        roi = np.reshape(roi, newshape=(546, 546, 1))
        roi = np.stack((roi,) * 3, axis=2)
        roi = np.reshape(roi, newshape=(546, 546, 3))
    upper_left = (730, 50)
    bottom_right = (1280, 600)
    cv_image[upper_left[1] + 2: bottom_right[1] - 2, upper_left[0] + 2: bottom_right[0] - 2] = roi
    return cv_image


def add_text_to_image(cv_image, text, org=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2, color=(255, 0, 0),
                      thickness=2):
    return cv2.putText(cv_image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
