HED_MODEL_URL = 'http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel'
HED_MODEL_FILEPATH = 'utils/hed_model/hed_pretrained_bsds.caffemodel'
HED_MODEL_PROTOTXT_FILEPATH = 'utils/hed_model/deploy.prototxt'

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_N_CHANNELS = 3
IMG_SHAPE = IMG_HEIGHT, IMG_WIDTH, IMG_N_CHANNELS
IMG_PATCH_HEIGHT = 16
IMG_PATCH_WIDTH = 16

USER_COLOR_POINTS_PER_IMG = 50
USER_COLOR_POINTS_CIRCLE_RADIUS = 3

TRAINING_DIR = 'data/training'
TRAINING_SOURCE_DIR = f'{TRAINING_DIR}/source'
TRAINING_TARGET_DIR = f'{TRAINING_DIR}/target'
TRAINING_BATCH_SIZE = 1

VALIDATION_DIR = 'data/validation'
VALIDATION_SOURCE_DIR = f'{VALIDATION_DIR}/source'
VALIDATION_TARGET_DIR = f'{VALIDATION_DIR}/target'
VALIDATION_BATCH_SIZE = 7
