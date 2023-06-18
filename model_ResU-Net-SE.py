from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, GlobalAveragePooling2D, Reshape, Multiply
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def squeeze_excitation_block(inputs, num_filters):
    se = GlobalAveragePooling2D()(inputs)
    se = Reshape((1, 1, num_filters))(se)
    se = Conv2D(num_filters // 16, 1, activation='relu', padding='same')(se)
    se = Conv2D(num_filters, 1, activation='sigmoid', padding='same')(se)
    se = Multiply()([inputs, se])
    return se

def encoder_block(inputs, num_filters):
    s = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(s)
    return s, p

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    x = squeeze_excitation_block(x, num_filters)
    return x

def build_residual_unet(input_shape):
    """ Input layer """
    inputs = Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    """ Bottleneck """
    b1 = conv_block(p4, 1024)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Extra Convolutional Layers """
    d4 = conv_block(d4, 64)
    d4 = conv_block(d4, 64)
    d4 = conv_block(d4, 64)

    """ Output layer """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="Residual_UNet_SqueezeExcitation")
    return model

if __name__ == "__main__":
    model = build_residual_unet((576, 576, 3))
    model.summary()
