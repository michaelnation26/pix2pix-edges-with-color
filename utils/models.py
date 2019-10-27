"""
Keras implementation of Pix2Pix from Jason Brownlee's tutorial.
https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
"""

from keras.initializers import RandomNormal
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU
from keras.models import Input, Model
from keras.optimizers import Adam


def get_discriminator_model(image_shape=(256,256,3)):
    kernel_weights_init = RandomNormal(stddev=0.02)
    input_src_image = Input(shape=image_shape)
    input_target_image = Input(shape=image_shape)

    # concatenate images channel-wise
    merged = Concatenate()([input_src_image, input_target_image])
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=kernel_weights_init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=kernel_weights_init)(d)
    patch_out = Activation('sigmoid')(d)

    # define model
    model = Model([input_src_image, input_target_image], patch_out, name='descriminator_model')
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])

    return model

def get_gan_model(g_model, d_model, image_shape=(256,256,3), L1_loss_lambda=100):
    """Combined generator and discriminator model. Used for updating the generator."""
    d_model.trainable = False
    input_src_image = Input(shape=image_shape)
    gen_out = g_model(input_src_image)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([input_src_image, gen_out])

    # src image as input, generated image and real/fake classification as output
    model = Model(input_src_image, [dis_out, gen_out], name='gan_model')
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, L1_loss_lambda])

    return model

def get_generator_model(image_shape=(256,256,3)):
    kernel_weights_init = RandomNormal(stddev=0.02)
    input_src_image = Input(shape=image_shape)

    # encoder model
    e1 = _encoder_block(input_src_image, 64, batchnorm=False)
    e2 = _encoder_block(e1, 128)
    e3 = _encoder_block(e2, 256)
    e4 = _encoder_block(e3, 512)
    e5 = _encoder_block(e4, 512)
    e6 = _encoder_block(e5, 512)
    e7 = _encoder_block(e6, 512)

    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(e7)
    b = Activation('relu')(b)

    # decoder model
    d1 = _decoder_block(b, e7, 512)
    d2 = _decoder_block(d1, e6, 512)
    d3 = _decoder_block(d2, e5, 512)
    d4 = _decoder_block(d3, e4, 512, dropout=False)
    d5 = _decoder_block(d4, e3, 256, dropout=False)
    d6 = _decoder_block(d5, e2, 128, dropout=False)
    d7 = _decoder_block(d6, e1, 64, dropout=False)

    # output
    g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(d7)
    out_image = Activation('tanh')(g)

    # define model
    model = Model(input_src_image, out_image, name='generator_model')

    return model

def _decoder_block(layer_in, skip_in, n_filters, dropout=True):
    kernel_weights_init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)

    return g

def _encoder_block(layer_in, n_filters, batchnorm=True):
    kernel_weights_init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_weights_init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)

    return g
