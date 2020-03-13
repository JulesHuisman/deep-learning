from keras.models import Model
from keras.layers import Dense, Input, Conv2D, add, Flatten, BatchNormalization, ReLU
from keras.optimizers import Adam
from keras.losses import mean_squared_error, binary_crossentropy
from keras.utils import plot_model

class ResBlock:
    """
    Custom class to create a residual block consisting of two convolution layers
    """
    def __new__(self, inputs, block):
        residual = inputs
    
        conv_1 = Conv2D(128, (3, 3), padding='same', name=f'res-conv-{block}-1')(inputs)
        norm_1 = BatchNormalization(name=f'res-norm-{block}-1')(conv_1)

        relu_1 = ReLU(name=f'res-relu-{block}-1')(norm_1)

        conv_2 = Conv2D(128, (3, 3), padding='same', name=f'res-conv-{block}-2')(relu_1)
        norm_2 = BatchNormalization(name=f'res-norm-{block}-2')(conv_2)

        out = add([residual, norm_2], name=f'res-add-{block}')

        out = ReLU(name=f'res-relu-{block}-2')(out)

        return out

class ConnectNet:
    """
    Network that plays connect four.
    It has two heads:
    
    1. Policy head (the value of all possible plays)
        [0.12935083, 0.13370915, 0.17631257, 0.13289174, 0.16338773, 0.14248851, 0.12185949]

    2. Value head (the change of winning)
        [0.58]
    """

    def __init__(self):
        self.model = self._model()

    def _model(self):
        board_input = Input(shape=(6, 7, 3), name='board')

        # Start conv block
        conv_1 = Conv2D(128, (3, 3), padding='same', name='conv-1')(board_input)
        norm_1 = BatchNormalization(name='norm-1')(conv_1)
        relu_1 = ReLU(name='relu-1')(norm_1)

        # Residual convolution blocks
        res_1 = ResBlock(relu_1, block=1)
        res_2 = ResBlock(res_1, block=2)
        res_3 = ResBlock(res_2, block=3)

        # Policy output
        policy_conv = Conv2D(128, (3, 3), padding='same', name='policy-conv')(res_3)
        policy_norm = BatchNormalization(name='policy-norm')(policy_conv)
        policy_relu = ReLU(name='policy-relu')(policy_norm)
        policy_flat = Flatten(name='policy-flat')(policy_relu)

        policy = Dense(7, name='policy', activation='softmax')(policy_flat)

        # Value output
        value_conv = Conv2D(128, (3, 3), padding='same', name='value-conv')(res_3)
        value_norm = BatchNormalization(name='value-norm')(value_conv)
        value_relu_1 = ReLU(name='value-relu-1')(value_norm)
        value_flat = Flatten(name='value-flat')(value_relu_1)

        value_dense = Dense(32, name='value-dense')(value_flat)
        value_relu_2 = ReLU(name='value-relu-2')(value_dense)

        value = Dense(1, name='value')(value_relu_2)

        model = Model(inputs=[board_input], outputs=[policy, value])

        model.compile(optimizer=Adam(0.001, 0.8, 0.999), loss={'value': 'mse', 'policy': 'categorical_crossentropy'})
        
        return model

    def plot(self):
        """Shows the structure of the network"""
        return plot_model(self.model, show_shapes=True, dpi=64)