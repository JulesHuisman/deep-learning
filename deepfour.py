import os
import matplotlib.pyplot as plt
import numpy as np
import mlflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ResBlock:
    """
    Custom class to create a residual block consisting of two convolution layers
    """
    def __new__(self, inputs, filters, kernel, l2_reg):
        from keras.layers import Conv2D, add, BatchNormalization, ReLU
        from keras.regularizers import l2

        residual = inputs
    
        conv_1 = Conv2D(filters,
                        (kernel, kernel),
                        padding='same',
                        data_format='channels_first',
                        kernel_regularizer=l2(l2_reg))(inputs)

        norm_1 = BatchNormalization(axis=1)(conv_1)
        relu_1 = ReLU()(norm_1)

        conv_2 = Conv2D(filters,
                        (kernel, kernel),
                        padding='same',
                        data_format='channels_first',
                        kernel_regularizer=l2(l2_reg))(relu_1)

        norm_2 = BatchNormalization(axis=1)(conv_2)
        out = add([residual, norm_2])
        out = ReLU()(out)

        return out

def objective_function_for_policy(y_true, y_pred):
    from keras import backend as K
    return K.sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1)

def objective_function_for_value(y_true, y_pred):
    from keras import backend as K
    from keras.losses import mean_squared_error
    return mean_squared_error(y_true, y_pred)

class DeepFour:
    """
    Network that plays connect four.
    It has two heads:
    
    1. Policy head (the value of all possible plays)
        [0.129, 0.133, 0.176, 0.132, 0.163, 0.142, 0.121]

    2. Value head (the chance of winning)
        [0.58]
    """

    def __init__(self, config, only_predict=False):
        self.config  = config

        # Create the model
        self.model = self._model()

        # Use model only for prediction
        if only_predict:
            from keras import backend
            backend.set_learning_phase(0)

    def _model(self):
        from keras.models import Model
        from keras.layers import Dense, Input, Conv2D, add, Flatten, BatchNormalization, ReLU
        from keras.optimizers import Adam, SGD
        from keras.losses import mean_squared_error, binary_crossentropy
        from keras.regularizers import l2
        from tensorflow.python.util import deprecation
        deprecation._PRINT_DEPRECATION_WARNINGS = False

        board_input = Input(shape=(2, 6, 7))

        # Start conv block
        conv_1 = Conv2D(self.config.n_filters,
                        (self.config.kernel, self.config.kernel),
                        padding='same',
                        data_format='channels_first',
                        kernel_regularizer=l2(self.config.l2_reg))(board_input)
        norm_1 = BatchNormalization(axis=1)(conv_1)
        relu_1 = ReLU()(norm_1)
        res = relu_1

        # Residual convolution blocks
        for _ in range(self.config.res_layers):
            res = ResBlock(res, self.config.n_filters, self.config.kernel, self.config.l2_reg)

        # Policy head
        policy_conv = Conv2D(2, (1, 1), data_format='channels_first', kernel_regularizer=l2(self.config.l2_reg))(res)
        policy_norm = BatchNormalization(axis=1)(policy_conv)
        policy_relu = ReLU()(policy_norm)
        policy_flat = Flatten()(policy_relu)

        # Policy output
        policy = Dense(7,
                       activation='softmax',
                       name='policy',
                       kernel_regularizer=l2(self.config.l2_reg))(policy_flat)

        # Value head
        value_conv   = Conv2D(1, (1, 1), data_format='channels_first', kernel_regularizer=l2(self.config.l2_reg))(res)
        value_norm   = BatchNormalization(axis=1)(value_conv)
        value_relu_1 = ReLU()(value_norm)
        value_flat   = Flatten()(value_relu_1)

        value_dense  = Dense(self.config.value_dense, kernel_regularizer=l2(self.config.l2_reg))(value_flat)
        value_relu_2 = ReLU()(value_dense)

        # Value output
        value = Dense(1,
                      activation='tanh',
                      name='value',
                      kernel_regularizer=l2(self.config.l2_reg))(value_relu_2)

        # Final model
        model = Model(inputs=[board_input], outputs=[policy, value])

        # Compile
        model.compile(optimizer=SGD(0.001, momentum=0.9),
                      loss={'value': objective_function_for_value, 'policy': objective_function_for_policy},
                      metrics={'value': [self.mean]})

        # Set the model name
        model._name = self.config.model
        
        return model

    def mean(self, y_true, y_pred):
        """Custom metric that returns the mean value"""
        from keras import backend as K
        return K.mean(y_pred)

    def predict(self, board):
        """Predict policy and value based on an encoded board"""
        policy, value = self.model.predict(np.array([board]), batch_size=1)
        policy, value = policy[0], value[0][0]

        return policy, value

    def plot(self):
        from keras.utils import plot_model
        """Shows the structure of the network"""
        return plot_model(self.model, show_shapes=True, dpi=64)

    def update_lr(self, total_steps):
        """
        Use a learning rate schedule to lower the learning rate over time
        https://github.com/Zeta36/connect4-alpha-zero/blob/master/src/connect4_zero/worker/optimize.py
        """
        import keras.backend as K

        # if total_steps < 500:
        #     lr = 0.01
        # elif total_steps < 2000:
        #     lr = 0.001
        # elif total_steps < 9000:
        #     lr = 0.0001
        # else:
        #     lr = 0.000025

        lr = 0.001
            
        mlflow.log_metric('learning-rate', lr, total_steps)
        K.set_value(self.model.optimizer.lr, lr)

    def load(self, postfix, log=True):
        """Load model weights"""
        try:
            self.model.load_weights(os.path.join('data', self.config.model, 'models', self.config.model + '.' + str(postfix) + '.h5'))
            self.version = postfix
            if log:
                print(f'Loaded network: \033[94m{self.config.model + "." + str(postfix)}\033[0m')
        except:
            if log:
                print('\033[93mModel not found\033[0m')

    def save(self, postfix):
        """
        Store model weights
        """
        storage_location = os.path.join('data', self.config.model, 'models')
        file_name = self.config.model + '.' + str(postfix) + '.h5'

        # Create a storage folder
        if not os.path.exists(storage_location):
            os.makedirs(storage_location)

        self.model.save_weights(os.path.join(storage_location, file_name))