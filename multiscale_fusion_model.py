
############## Multi-Scale Fusion Model #############
############# WHat's There In The Dark ##############
`````############# Sauradip Nag ###############

from keras.layers import Input, Conv2D, concatenate, Flatten, Dense , Conv1D
from keras.models import Model
import numpy as np


def build_model(inp_shape):
    # make the model
    inp1 = Input(shape=inp_shape, name='patch1')
    inp2 = Input(shape=inp_shape, name='patch2')

    conv1_5 = Conv2D(128, (5, 5), activation='tanh', padding='valid')
    conv2_5 = Conv2D(64, (5, 5), activation='tanh', padding='valid')
    conv3_5 = Conv2D(32, (5, 5), activation='tanh', padding='valid')
    conv1d_ch1 = Conv1D(32, (5,5), strides=1, padding='valid')

    conv1_7 = Conv2D(128, (7, 7), activation='tanh', padding='valid')
    conv2_7 = Conv2D(64, (7, 7), activation='tanh', padding='valid')
    conv3_7 = Conv2D(32, (7, 7), activation='tanh', padding='valid')
    conv1d_ch2 = Conv1D(32, (7,7), strides=1, padding='valid')

############ For scale 5 X 5 ############## 

    x1 = conv1_5(inp1)
    x1 = conv2_5(x1)
    x1 = conv3_5(x1)

    x2 = conv1_5(inp2)
    x2 = conv2_5(x2)
    x2 = conv3_5(x2)
    x_ch1 = concatenate([x1, x2], axis=1)
    ch1_final = conv1d_ch1(x_ch1)

############ For scale 7 X 7 ############## 

    x3 = conv1_7(inp1)
    x3 = conv2_7(x3)
    x3 = conv3_7(x3)

    x4 = conv1_7(inp2)
    x4 = conv2_7(x4)
    x4 = conv3_7(x4)
    x_ch2 = concatenate([x3, x4], axis=1)
    ch2_final = conv1d_ch2(x_ch2)
    
    x = concatenate([ch1_final, ch2_final], axis=1)
    # x = concatenate([ch2_final, ch1_final], axis=1)
    x = Flatten()(x)
    x = Dense(512, activation='tanh')(x)
    x = Dense(128, activation='tanh')(x)
    output = Dense(2, activation='softmax', name='output')(x)

    model = Model(inputs=[inp1, inp2], outputs=output)

    return model

