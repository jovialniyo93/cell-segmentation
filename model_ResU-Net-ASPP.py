from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Input, DepthwiseConv2D
from tensorflow.keras.models import Model

def batchnorm_relu(inputs):
    """ Batch Normalization & ReLU """
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x

def residual_block(inputs, num_filters, strides=1):
    """ Convolutional Layers """
    x = batchnorm_relu(inputs)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=1)(x)

    """ Shortcut Connection (Identity Mapping) """
    s = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)

    """ Addition """
    x = x + s
    return x

def atrous_spatial_pyramid_pooling(inputs, num_filters):
    """ Atrous Spatial Pyramid Pooling """
    pool1 = DepthwiseConv2D(3, padding='same', dilation_rate=1)(inputs)
    pool6 = DepthwiseConv2D(3, padding='same', dilation_rate=6)(inputs)
    pool12 = DepthwiseConv2D(3, padding='same', dilation_rate=12)(inputs)
    pool18 = DepthwiseConv2D(3, padding='same', dilation_rate=18)(inputs)

    x = Concatenate()([pool1, pool6, pool12, pool18])
    return x

def decoder_block(inputs, skip_features, num_filters):
    """ Decoder Block """
    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip_features])
    x = residual_block(x, num_filters, strides=1)
    return x

def build_aspp_resunet(input_shape):
    """ ASPP-ResU-Net Architecture """
    inputs = Input(input_shape)

    """ Endoder 1 """
    x = Conv2D(64, 3, padding="same", strides=1)(inputs)
    x = batchnorm_relu(x)
    x = Conv2D(64, 3, padding="same", strides=1)(x)
    s = Conv2D(64, 1, padding="same")(inputs)
    s1 = x + s

    """ Encoder 2, 3, 4 """
    s2 = residual_block(s1, 128, strides=2)
    s3 = residual_block(s2, 256, strides=2)
    s4 = residual_block(s3, 512, strides=2)

    """ Bridge """
    b = residual_block(s4, 1024, strides=2)

    """ ASPP Block """
    a = atrous_spatial_pyramid_pooling(b, 256)

    """ Decoder 1, 2, 3, 4 """
    x = decoder_block(a, s4, 512)
    x = decoder_block(x, s3, 256)
    x = decoder_block(x, s2, 128)
    x = decoder_block(x, s1, 64)

    """ Classifier """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    """ Model """
    model = Model(inputs, outputs, name="ASPP-ResU-Net")

    return model

if __name__ == "__main__":
    shape = (576, 576, 3)
    model = build_aspp_resunet(shape)
    model.summary()
