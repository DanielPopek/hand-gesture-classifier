import numpy as np
import tensorflow as tf


def predict_image(model, image, is_color_image, img_width, img_height, class_names):
    img = np.array(image)  # 496x496
    print('image: ', img.shape)
    if not is_color_image:
        img = img[:, :, None] * np.ones(3, dtype=int)[None, None, :]
    print('image: ', img.shape)
    image_resized = tf.image.resize(
        img,
        size=(img_width, img_height),
        method=tf.image.ResizeMethod.BILINEAR,
        preserve_aspect_ratio=False,
        antialias=False,
        name=None
    )
    print('image_resized: ', image_resized.shape)
    single_img = np.reshape(image_resized, newshape=(1, 54, 54, 3))
    model_prediction = model.predict(single_img)[0]
    return class_names[model_prediction]
