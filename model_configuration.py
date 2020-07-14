from model_type import ModelType

MODEL_FILENAMES_DIST = {
    ModelType.CNN_3D: 'weights/WEIGHTS_CNN_MASK_ALL_DATA.npy',
    ModelType.CNN_BIN: 'weights/WEIGHTS_CNN_MASK_ALL_DATA.npy',
    ModelType.CNN_MASK: 'weights/WEIGHTS_CNN_MASK_ALL_DATA.npy',
    ModelType.VGG_3D: 'weights/WEIGHTS_VGG_MASK_ALL_DATA.npy',
    ModelType.VGG_BIN: 'weights/WEIGHTS_VGG_MASK_ALL_DATA.npy',
    ModelType.VGG_MASK: 'weights/WEIGHTS_VGG_MASK_ALL_DATA.npy'
}


def get_model_filename(model_type: ModelType):
    return MODEL_FILENAMES_DIST[model_type]
