from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Input, GlobalAveragePooling2D, Reshape, multiply, Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


def batchnorm_relu(inputs):
    """ Batch Normalization & ReLU """
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x

def attention_block(inputs, num_filters):
    """ Attention Mechanism """
    skip = Conv2D(num_filters, 1, padding="same")(inputs)
    attention = GlobalAveragePooling2D()(skip)
    attention = Reshape((1, 1, num_filters))(attention)
    attention = Conv2D(num_filters, 1, padding="same", activation="sigmoid")(attention)
    attention = multiply([skip, attention])
    return attention

def se_block(inputs, num_filters):
    """ Squeeze and Excitation (SE) Block """
    se = GlobalAveragePooling2D()(inputs)
    se = Reshape((1, 1, num_filters))(se)
    se = Conv2D(num_filters // 16, 1, padding="same", activation="relu")(se)
    se = Conv2D(num_filters, 1, padding="same", activation="sigmoid")(se)
    se = multiply([inputs, se])
    return se

def aspp_block(inputs, num_filters):
    """ Atrous Spatial Pyramid Pooling (ASPP) Block """
    rate1 = 1
    rate2 = 2
    rate3 = 3
    rate4 = 4

    # Branch 1
    b1 = Conv2D(num_filters, 1, padding="same", activation="relu")(inputs)

    # Branch 2
    b2 = Conv2D(num_filters, 3, padding="same", dilation_rate=rate1, activation="relu")(inputs)

    # Branch 3
    b3 = Conv2D(num_filters, 3, padding="same", dilation_rate=rate2, activation="relu")(inputs)

    # Branch 4
    b4 = Conv2D(num_filters, 3, padding="same", dilation_rate=rate3, activation="relu")(inputs)

    # Branch 5
    b5 = Conv2D(num_filters, 3, padding="same", dilation_rate=rate4, activation="relu")(inputs)

    # Concatenate branches
    concatenated = Concatenate()([b1, b2, b3, b4, b5])

    return concatenated

def residual_block(inputs, num_filters, strides=1):
    """ Convolutional Layers """
    x = batchnorm_relu(inputs)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=1)(x)

    """ Shortcut Connection (Identity Mapping) """
    s = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)

    """ Attention Mechanism """
    attention = attention_block(x, num_filters)

    """ Squeeze and Excitation (SE) Block """
    se = se_block(x, num_filters)

    """ Addition """
    x = Add()([x, attention, se, s])
    return x

def decoder_block(inputs, skip_features, num_filters):
    """ Decoder Block """
    x = UpSampling2D((2, 2))(inputs)

    # Adjust the spatial dimensions of skip_features to match x
    skip_features = Conv2D(num_filters, 1, padding="same")(skip_features)
    skip_features = UpSampling2D((2, 2))(skip_features)

    x = Concatenate()([x, skip_features])
    x = residual_block(x, num_filters, strides=1)
    return x



def build_attention_resunet(input_shape):
    """ RESUNET Architecture with Attention, SE, and ASPP """
    inputs = Input(input_shape)

    """ Endoder 1 """
    x = Conv2D(64, 3, padding="same", strides=1)(inputs)
    x = batchnorm_relu(x)
    x = Conv2D(64, 3, padding="same", strides=1)(x)
    s = Conv2D(64, 1, padding="same")(inputs)
    s1 = Add()([x, s])

    """ Encoder 2, 3, 4 """
    s2 = residual_block(s1, 128, strides=2)
    s3 = residual_block(s2, 256, strides=2)
    s4 = residual_block(s3, 512, strides=2)

    """ ASPP """
    b = aspp_block(s4, 1024)

    """ Decoder 1, 2, 3, 4 """
    x = decoder_block(b, s4, 512)
    x = decoder_block(x, s3, 256)
    x = decoder_block(x, s2, 128)
    x = decoder_block(x, s1, 64)

    """ Classifier """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    """ Model """
    model = Model(inputs, outputs, name="AttentionResidualUNET")
    return model

if __name__ == "__main__":
    shape = (576, 576, 3)
    model = build_attention_resunet(shape)
    model.summary()
