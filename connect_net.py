import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Model
from keras.layers import Dense, Input, Conv2D, add, Flatten, BatchNormalization, ReLU
from keras.optimizers import Adam
from keras.losses import mean_squared_error, binary_crossentropy
from keras.utils import plot_model
from keras.regularizers import l2
import tensorflow as tf

class ResBlock:
    """
    Custom class to create a residual block consisting of two convolution layers
    """
    def __new__(self, inputs, filters, l2_reg):
        residual = inputs
    
        conv_1 = Conv2D(filters,
                        (4, 4),
                        padding='same',
                        use_bias=False,
                        kernel_regularizer=l2(l2_reg))(inputs)

        norm_1 = BatchNormalization()(conv_1)
        relu_1 = ReLU()(norm_1)

        conv_2 = Conv2D(filters,
                        (4, 4),
                        padding='same',
                        use_bias=False,
                        kernel_regularizer=l2(l2_reg))(relu_1)

        norm_2 = BatchNormalization()(conv_2)
        out = add([residual, norm_2])
        out = ReLU()(out)

        return out

def softmax_cross_entropy_with_logits(y_true, y_pred):
    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

    return loss

class ConnectNet:
    """
    Network that plays connect four.
    It has two heads:
    
    1. Policy head (the value of all possible plays)
        [0.12935083, 0.13370915, 0.17631257, 0.13289174, 0.16338773, 0.14248851, 0.12185949]

    2. Value head (the change of winning)
        [0.58]
    """

    def __init__(self, name, filters=75, l2_reg=0.0001):
        self.name    = name
        self.filters = filters
        self.l2_reg  = l2_reg

        # Create the model
        self.model   = self._model()

    def _model(self):
        board_input = Input(shape=(6, 7, 3))

        # Start conv block
        conv_1 = Conv2D(self.filters, (4, 4), padding='same', kernel_regularizer=l2(self.l2_reg))(board_input)
        norm_1 = BatchNormalization()(conv_1)
        relu_1 = ReLU()(norm_1)

        # Residual convolution blocks
        res_1 = ResBlock(relu_1, self.filters, self.l2_reg)
        res_2 = ResBlock(res_1, self.filters, self.l2_reg)
        res_3 = ResBlock(res_2, self.filters, self.l2_reg)
        res_4 = ResBlock(res_3, self.filters, self.l2_reg)
        res_5 = ResBlock(res_4, self.filters, self.l2_reg)

        # Policy head
        policy_conv = Conv2D(2, (1, 1), use_bias=False, kernel_regularizer=l2(self.l2_reg))(res_5)
        policy_norm = BatchNormalization()(policy_conv)
        policy_relu = ReLU()(policy_norm)
        policy_flat = Flatten()(policy_relu)

        # Policy output
        policy = Dense(7,
                       activation='softmax',
                       name='policy',
                       use_bias=False,
                       kernel_regularizer=l2(self.l2_reg))(policy_flat)

        # Value head
        value_conv   = Conv2D(1, (1, 1), use_bias=False, kernel_regularizer=l2(self.l2_reg))(res_5)
        value_norm   = BatchNormalization()(value_conv)
        value_relu_1 = ReLU()(value_norm)
        value_flat   = Flatten()(value_relu_1)

        value_dense  = Dense(32, use_bias=False, kernel_regularizer=l2(self.l2_reg))(value_flat)
        value_relu_2 = ReLU()(value_dense)

        # Value output
        value = Dense(1,
                      activation='tanh',
                      name='value',
                      use_bias=False,
                      kernel_regularizer=l2(self.l2_reg))(value_relu_2)

        # Final model
        model = Model(inputs=[board_input], outputs=[policy, value])

        # Compile
        model.compile(optimizer=Adam(0.001, 0.8, 0.999), loss={'value': 'mse', 'policy': softmax_cross_entropy_with_logits})

        # Set the model name
        model.name = self.name
        
        return model

    def plot(self):
        """Shows the structure of the network"""
        return plot_model(self.model, show_shapes=True, dpi=64)

    def load(self, postfix):
        """Load model weights"""
        self.model.load_weights(os.path.join('data', self.name, 'models', self.name + '-' + str(postfix) + '.h5'))

    def save(self, postfix):
        """Store model weights"""
        self.model.save_weights(os.path.join('data', self.name, 'models', self.name + '-' + str(postfix) + '.h5'))