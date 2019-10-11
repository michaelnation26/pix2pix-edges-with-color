GAN_L1_LOSS_LAMBDA = 100

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_N_CHANNELS = 3
IMG_SHAPE = IMG_HEIGHT, IMG_WIDTH, IMG_N_CHANNELS
IMG_PATCH_HEIGHT = 16
IMG_PATCH_WIDTH = 16

USER_COLOR_POINTS_PER_IMG = 350
USER_COLOR_POINTS_RADIUS = 2

ZAPPOS_DATASET_URL = 'http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images-square.zip'
ZAPPOS_DATASET_NAME = 'ut-zap50k-images-square'
ZAPPOS_DATASET_SNEAKERS_DIR = f'{ZAPPOS_DATASET_NAME}/Shoes/Sneakers and Athletic Shoes'
ZAPPOS_DATASET_MIN_SHOE_MODEL_ID_COUNT = 2

TRAINING_DIR = 'data/training'
TRAINING_SOURCE_DIR = f'{TRAINING_DIR}/source'
TRAINING_TARGET_DIR = f'{TRAINING_DIR}/target'
TRAINING_BATCH_SIZE = 4

VALIDATION_DIR = 'data/validation'
VALIDATION_SOURCE_DIR = f'{VALIDATION_DIR}/source'
VALIDATION_TARGET_DIR = f'{VALIDATION_DIR}/target'
VALIDATION_BATCH_SIZE = 6

PREPROCESSED_DATASET_URL = 'https://github.com/michaelnation26/pix2pix-edges-with-color/releases/download/v1.0/data.zip'
TRAINED_GENERATOR_MODEL_URL = 'https://github.com/michaelnation26/pix2pix-edges-with-color/releases/download/v1.0/gen_model.h5'
