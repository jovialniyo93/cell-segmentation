from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def attention_block(inputs, skip_features, num_filters):
    g = Conv2D(num_filters, 1, padding="same")(skip_features)
    x = Conv2D(num_filters, 1, padding="same")(inputs)

    theta_x = Conv2D(num_filters, 1, padding="same")(x)
    phi_g = Conv2D(num_filters, 1, padding="same")(g)

    theta_x = BatchNormalization()(theta_x)
    phi_g = BatchNormalization()(phi_g)

    f = Activation("relu")(theta_x + phi_g)
    psi_f = Conv2D(1, 1, padding="same")(f)
    psi_f = Activation("sigmoid")(psi_f)

    rate = psi_f * x
    attention_output = Concatenate()([rate, skip_features])

    return attention_output

def encoder_block(inputs, num_filters):
    s = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(s)
    return s, p

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    x = attention_block(x, skip_features, num_filters)
    return x

def build_residual_attention_unet(input_shape):
    """ Input layer """
    inputs = Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    """ Bottleneck """
    b1 = conv_block(p4, 1024)
    b1 = attention_block(b1, p4, 1024)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Output layer """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="Residual_Attention_UNet")
    return model

if __name__ == "__main__":
    model = build_residual_attention_unet((576, 576, 3))
    model.summary()
