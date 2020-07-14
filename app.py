import cv2
import time
import os
from enum import IntEnum
from model_cnn import recreate_cnn_model_from_weights
from model_vgg import recreate_vgg_model_from_weights
from model_configuration import get_model_filename
from model_type import ModelType
from gesture_predictor import predict_image
import image_tool
from text_shower import TextShower
from label_sequence_analyser import LabelSequenceAnalyser


class ProgramMode(IntEnum):
    SCRAP_CLASSIC = 1
    PREDICT_VGG_3D = 2
    PREDICT_VGG_BIN = 3
    PREDICT_CNN_3D = 4
    PREDICT_CNN_BIN = 5
    PREDICT_CNN_MASK = 6
    PREDICT_VGG_MASK = 7
    SCRAP_WITH_MASK = 8

    def is_predict_mode(self):
        return self in [ProgramMode.PREDICT_VGG_3D, ProgramMode.PREDICT_VGG_BIN, ProgramMode.PREDICT_CNN_3D,
                        ProgramMode.PREDICT_CNN_BIN, ProgramMode.PREDICT_CNN_MASK, ProgramMode.PREDICT_VGG_MASK]

    def is_scrap_mode(self):
        return self in [ProgramMode.SCRAP_CLASSIC, ProgramMode.SCRAP_WITH_MASK]

    def is_mode_with_mask(self):
        return self in [ProgramMode.SCRAP_WITH_MASK, ProgramMode.PREDICT_VGG_MASK, ProgramMode.PREDICT_CNN_MASK]


class PreviewMode(IntEnum):
    COLOR = 1
    NAIVE_BINARY = 2
    MASK_BINARY = 3

    def get_next(self):
        return {
            PreviewMode.COLOR: PreviewMode.NAIVE_BINARY,
            PreviewMode.NAIVE_BINARY: PreviewMode.MASK_BINARY,
            PreviewMode.MASK_BINARY: PreviewMode.COLOR,
        }[self]


BASE_DIR = 'sessions'
FINAL_DIR = 'final'
FIST_DIR = 'fist'
OK_DIR = 'ok'
VICTORIA_DIR = 'victoria'
PALM_DIR = 'palm'
ARROW_DIR = 'arrow'
THREE_DIR = 'three'
SUB_DIRS = [FIST_DIR, OK_DIR, VICTORIA_DIR, PALM_DIR, ARROW_DIR, THREE_DIR]

ARROW_KEY = ord('a')
OK_KEY = ord('o')
VICTORIA_KEY = ord('v')
PALM_KEY = ord('p')
THREE_KEY = ord('t')
FIST_KEY = ord('f')
SCRAP_KEYS = [ARROW_KEY, OK_KEY, VICTORIA_KEY, PALM_KEY, THREE_KEY, FIST_KEY]
SCRAP_DIR_DICT = {
    ARROW_KEY: ARROW_DIR,
    VICTORIA_KEY: VICTORIA_DIR,
    PALM_KEY: PALM_DIR,
    THREE_KEY: THREE_DIR,
    FIST_KEY: FIST_DIR,
    OK_KEY: OK_DIR
}

SCRAP_DATA_MODE_KEY = ord('1')
PREDICT_VGG_3D_MODE_KEY = ord('2')
PREDICT_VGG_BIN_MODE_KEY = ord('3')
PREDICT_CNN_3D_MODE_KEY = ord('4')
PREDICT_CNN_BIN_MODE_KEY = ord('5')
PREDICT_VGG_MASK_MODE_KEY = ord('6')
PREDICT_CNN_MASK_MODE_KEY = ord('7')
SCRAP_MASK_DATA_MODE_KEY = ord('8')

MODE_KEYS = [SCRAP_DATA_MODE_KEY, PREDICT_VGG_3D_MODE_KEY, PREDICT_VGG_BIN_MODE_KEY,
             PREDICT_CNN_3D_MODE_KEY, PREDICT_CNN_BIN_MODE_KEY, PREDICT_VGG_MASK_MODE_KEY,
             PREDICT_CNN_MASK_MODE_KEY, SCRAP_MASK_DATA_MODE_KEY]
MODE_DICT = {
    SCRAP_DATA_MODE_KEY: ProgramMode.SCRAP_CLASSIC,
    PREDICT_VGG_3D_MODE_KEY: ProgramMode.PREDICT_VGG_3D,
    PREDICT_VGG_BIN_MODE_KEY: ProgramMode.PREDICT_VGG_BIN,
    PREDICT_CNN_3D_MODE_KEY: ProgramMode.PREDICT_CNN_3D,
    PREDICT_CNN_BIN_MODE_KEY: ProgramMode.PREDICT_CNN_BIN,
    PREDICT_VGG_MASK_MODE_KEY: ProgramMode.PREDICT_VGG_MASK,
    PREDICT_CNN_MASK_MODE_KEY: ProgramMode.PREDICT_CNN_MASK,
    SCRAP_MASK_DATA_MODE_KEY: ProgramMode.SCRAP_WITH_MASK
}

ESCAPE_KEY = 27  # classic escape button
CHANGE_PREVIEW_MODE_KEY = ord('0')

BASE_TEXT_SHOWER = TextShower()
LABEL_SEQUENCE_ANALYSER = LabelSequenceAnalyser()

MODEL_VGG_3D = recreate_vgg_model_from_weights(get_model_filename(ModelType.VGG_3D))
MODEL_VGG_BIN = recreate_vgg_model_from_weights(get_model_filename(ModelType.VGG_BIN))
MODEL_CNN_3D = recreate_cnn_model_from_weights(get_model_filename(ModelType.CNN_3D))
MODEL_CNN_BIN = recreate_cnn_model_from_weights(get_model_filename(ModelType.CNN_BIN))
MODEL_VGG_MASK = recreate_cnn_model_from_weights(get_model_filename(ModelType.CNN_BIN))
MODEL_CNN_MASK = recreate_cnn_model_from_weights(get_model_filename(ModelType.CNN_BIN))

MODE_MODEL_DICT = {
    ProgramMode.PREDICT_VGG_3D: MODEL_VGG_3D,
    ProgramMode.PREDICT_VGG_BIN: MODEL_VGG_BIN,
    ProgramMode.PREDICT_VGG_MASK: MODEL_VGG_MASK,
    ProgramMode.PREDICT_CNN_3D: MODEL_CNN_3D,
    ProgramMode.PREDICT_CNN_BIN: MODEL_CNN_BIN,
    ProgramMode.PREDICT_CNN_MASK: MODEL_CNN_MASK
}

CLASS_NAMES = ['arrow', 'fist', 'ok', 'palm', 'three', 'victoria']


def create_session_dirs(session_name):
    def create_base_structure():
        system_dir = os.getcwd()
        base_dir = system_dir + '/' + BASE_DIR
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

    def create_session_structure():
        system_dir = os.getcwd()
        base_dir = system_dir + '/' + BASE_DIR
        session_dir = base_dir + '/' + session_name
        if not os.path.exists(session_dir):
            os.mkdir(session_dir)
        for gesture in SUB_DIRS:
            if not os.path.exists(session_dir + '/' + gesture):
                os.mkdir(session_dir + '/' + gesture)

    create_base_structure()
    create_session_structure()
    return


def support_region_of_interest(cv_image_frame):
    upper_left = (730, 50)
    bottom_right = (1280, 600)

    # Rectangle marker
    _ = cv2.rectangle(cv_image_frame, upper_left, bottom_right, (100, 50, 200), 2)
    rect_img = cv_image_frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
    inside_rect = cv_image_frame[upper_left[1] + 2: bottom_right[1] - 2, upper_left[0] + 2: bottom_right[0] - 2]

    sketcher_rect = rect_img

    # Replacing the sketched image on Region of Interest
    cv_image_frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]] = sketcher_rect
    return cv_image_frame, inside_rect


def support_scrap_mode(scrap_key, image_roi, session_dir, scrap_mode, bg_model):
    image_to_save = image_tool.background_threshold(image_roi, bg_model) \
        if scrap_mode is ProgramMode.SCRAP_WITH_MASK else image_roi
    subdir_name = SCRAP_DIR_DICT[scrap_key]
    cv2.imwrite(session_dir + '/' + subdir_name + '/' + str(time.time()) + '.jpg', image_to_save)
    return


def support_predict(image_roi, bg_model, selected_mode):
    model = MODE_MODEL_DICT[selected_mode]
    is_color_image = selected_mode in [ProgramMode.PREDICT_VGG_3D, ProgramMode.PREDICT_CNN_3D]
    print(selected_mode, is_color_image)
    image = None

    if selected_mode in [ProgramMode.PREDICT_CNN_3D, ProgramMode.PREDICT_VGG_3D]:
        image = image_roi
    elif selected_mode in [ProgramMode.PREDICT_CNN_BIN, ProgramMode.PREDICT_VGG_BIN]:
        image = image_tool.naive_threshold(image_roi)
    elif selected_mode in [ProgramMode.PREDICT_CNN_MASK, ProgramMode.PREDICT_VGG_MASK]:
        image = image_tool.background_threshold(image_roi, bg_model)

    label = predict_image(model, image, is_color_image, 54, 54, CLASS_NAMES)
    LABEL_SEQUENCE_ANALYSER.put_label(label)
    label_to_show = LABEL_SEQUENCE_ANALYSER.get_label()
    if label_to_show is not None:
        BASE_TEXT_SHOWER.set_new_text(label_to_show, 1000)
    return


def support_mode_select(pressed_mode_key):
    selected_mode = MODE_DICT[pressed_mode_key]
    print('Activate', selected_mode)
    return selected_mode


def support_label_to_show(image):
    return image_tool.add_text_to_image(image, BASE_TEXT_SHOWER.text) if BASE_TEXT_SHOWER.is_to_show() else image


def get_roi_transformed_image(roi_image, preview_mode, bg_model):
    if preview_mode == PreviewMode.COLOR:
        return roi_image
    elif preview_mode == PreviewMode.MASK_BINARY:
        return image_tool.background_threshold(roi_image, bg_model)
    elif preview_mode == PreviewMode.NAIVE_BINARY:
        return image_tool.naive_threshold(roi_image)


def run_modes(session_name, mirror=False):
    system_dir = os.getcwd()
    base_dir = system_dir + '/' + BASE_DIR
    session_dir = base_dir + '/' + session_name
    cam = cv2.VideoCapture(0)

    selected_mode = ProgramMode.SCRAP_CLASSIC
    preview_mode = PreviewMode.COLOR

    bg_sub_threshold = 50

    bg_model = None

    while True:
        ret_val, image_frame = cam.read()
        if mirror:
            image_frame = cv2.flip(image_frame, 1)

        image_frame, inside_rect = support_region_of_interest(image_frame)

        pressed_key = cv2.waitKey(1)

        # press to change mode
        if pressed_key == ESCAPE_KEY:
            break
        elif pressed_key == CHANGE_PREVIEW_MODE_KEY:
            preview_mode = preview_mode.get_next()
            if preview_mode == PreviewMode.MASK_BINARY and bg_model is None:
                bg_model = cv2.createBackgroundSubtractorMOG2(0, bg_sub_threshold)
        elif pressed_key in MODE_KEYS:
            selected_mode = support_mode_select(pressed_key)
            if selected_mode.is_mode_with_mask():
                bg_model = cv2.createBackgroundSubtractorMOG2(0, bg_sub_threshold)

        # support modes
        if selected_mode.is_scrap_mode() and pressed_key in SCRAP_KEYS:
            support_scrap_mode(pressed_key, inside_rect, session_dir, selected_mode, bg_model)
        elif selected_mode.is_predict_mode():
            support_predict(inside_rect, bg_model, selected_mode)

        inside_rect = get_roi_transformed_image(inside_rect, preview_mode, bg_model)
        image_frame = image_tool.place_roi_in_whole_img(image_frame, inside_rect)
        image_frame = image_tool.add_text_to_image(image_frame, str(selected_mode), org=(50, 690))
        image_frame = support_label_to_show(image_frame)

        cv2.imshow('data scarper', image_frame)

    cv2.destroyAllWindows()
    return


def run_preview(session_name):
    create_session_dirs(session_name)
    run_modes(session_name, mirror=True)
    return


run_preview('test')
