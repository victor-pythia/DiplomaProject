import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers import MaxPool2D, UpSampling2D, Lambda, Multiply
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.initializers import random_normal
from keras.regularizers import l2
from ModelApi import ModelApi
from logging import getLogger
from tensorflow import Session
from keras.backend import set_session
import hashlib
import json
from tensorflow.compat.v1.logging import set_verbosity, ERROR
from tensorflow.compat.v1.math import log_softmax

set_verbosity(ERROR)

logger = getLogger(__name__)


def logsoftmax(x):
    return logsoftmax(x)

class ActorCritic:
    def __init__(self, config, sess=None):
        self.config = config
        self.model = None
        self.api = None
        self.session = sess or Session
        set_session(self.session)

    def creeaza(self):
        mc = self.config.model
        in_x = x = Input((mc.dim_input, 8, 8))
        x = Conv2D(filters=mc.nr_filtre, kernel_size=mc.dim_primul_kernel, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="input_conv-" + str(mc.dim_primul_kernel) + "-" + str(mc.nr_filtre))(x)
        x = BatchNormalization(axis=-1,
                               name="input_batchnorm")(x)
        x = PReLU(random_normal(), alpha_regularizer='l2')(x)

        for i in range(mc.nr_bloc_rezid):
            x = self.adauga_bloc_rezidual(x, i + 1)

        res_out = x

        # output strategie
        x = Conv2D(filters=32, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(mc.l2_reg),
                   name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=-1, name="policy_batchnorm")(x)
        x = PReLU(random_normal(), alpha_regularizer='l2')(x)
        x = Flatten(name="policy_flatten")(x)

        policy_out = Dense(self.config.nr_etichete, kernel_regularizer=l2(mc.l2_reg), activation='softmax',
                           name="strategie")(x)
        #policy_out = Activation(logsoftmax)(policy_out)

        # output valoare
        x = Conv2D(filters=32, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(mc.l2_reg),
                   name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=-1, name="value_batchnorm")(x)
        x = PReLU(random_normal(), alpha_regularizer='l2')(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.nr_fc_valoare, kernel_regularizer=l2(mc.l2_reg), activation=None, name="value_dense")(x)
        x = PReLU(random_normal(), alpha_regularizer='l2')(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="valoare")(x)

        self.model = Model(in_x, [policy_out, value_out], name="model_sah")
        if mc.print_summary:
            self.model.summary()

    def adauga_bloc_rezidual(self, x, index):
        mc = self.config.model
        in_x = x
        nume_bloc = "res" + str(index)
        x = Conv2D(filters=mc.nr_filtre, kernel_size=mc.dim_kernel, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=nume_bloc + "_conv1-" + str(mc.dim_kernel) + "-" + str(mc.nr_filtre))(x)
        x = BatchNormalization(axis=-1, name=nume_bloc + "_batchnorm1")(x)
        x = PReLU(random_normal(), alpha_regularizer='l2')(x)
        x = Conv2D(filters=mc.nr_filtre, kernel_size=mc.dim_kernel, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=nume_bloc + "_conv2-" + str(mc.dim_kernel) + "-" + str(mc.nr_filtre))(x)
        x = BatchNormalization(axis=-1, name="res" + str(index) + "_batchnorm2")(x)
        x = Add(name=nume_bloc + "_add")([in_x, x])
        x = PReLU(random_normal(), alpha_regularizer='l2')(x)
        return x

    # def attention_block(self, input, input_channels=None, output_channels=None, encoder_depth=1):
    #     """
    #     attention block
    #     https://arxiv.org/abs/1704.06904
    #     """
    #
    #     p = 1
    #     t = 2
    #     r = 1
    #
    #     if input_channels is None:
    #         input_channels = input.get_shape()[-1].value
    #     if output_channels is None:
    #         output_channels = input_channels
    #
    #     # First Residual Block
    #     for i in range(p):
    #         input = self.adauga_bloc_rezidual(input)
    #
    #     # Trunc Branch
    #     output_trunk = input
    #     for i in range(t):
    #         output_trunk = self.adauga_bloc_rezidual(output_trunk)
    #
    #     # Soft Mask Branch
    #
    #     ## encoder
    #     ### first down sampling
    #     output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    #     for i in range(r):
    #         output_soft_mask = self.adauga_bloc_rezidual(output_soft_mask)
    #
    #     skip_connections = []
    #     for i in range(encoder_depth - 1):
    #
    #         ## skip connections
    #         output_skip_connection = self.adauga_bloc_rezidual(output_soft_mask)
    #         skip_connections.append(output_skip_connection)
    #         # print ('skip shape:', output_skip_connection.get_shape())
    #
    #         ## down sampling
    #         output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
    #         for _ in range(r):
    #             output_soft_mask = self.adauga_bloc_rezidual(output_soft_mask)
    #
    #             ## decoder
    #     skip_connections = list(reversed(skip_connections))
    #     for i in range(encoder_depth - 1):
    #         ## upsampling
    #         for _ in range(r):
    #             output_soft_mask = self.adauga_bloc_rezidual(output_soft_mask)
    #         output_soft_mask = UpSampling2D()(output_soft_mask)
    #         ## skip connections
    #         output_soft_mask = Add()([output_soft_mask, skip_connections[i]])
    #
    #     ### last upsampling
    #     for i in range(r):
    #         output_soft_mask = self.adauga_bloc_rezidual(output_soft_mask)
    #     output_soft_mask = UpSampling2D()(output_soft_mask)
    #
    #     ## Output
    #     output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    #     output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    #     output_soft_mask = Activation('sigmoid')(output_soft_mask)
    #
    #     # Attention: (1 + output_soft_mask) * output_trunk
    #     output = Lambda(lambda x: x + 1)(output_soft_mask)
    #     output = Multiply()([output, output_trunk])  #
    #
    #     # Last Residual Block
    #     for i in range(p):
    #         output = self.adauga_bloc_rezidual(output)
    #
    #     return output

    def get_pipes(self, num=1):
        if self.api is None:
            self.api = ModelApi(self)
            self.api.start()
        return [self.api.create_pipe() for _ in range(num)]

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def incarca(self, config_path, weight_path):
        with self.session.as_default():
            with self.session.graph.as_default():
                set_session(self.session)
                if os.path.exists(config_path) and os.path.exists(weight_path):
                    logger.debug(f"Incarc modelul aflat in {config_path}")
                    with open(config_path, "rt") as f:
                        self.model = Model.from_config(json.load(f))
                    self.model.load_weights(weight_path)
                    self.model._make_predict_function()
                    self.digest = self.fetch_digest(weight_path)
                    logger.debug(f"Incarc modelul cu codul = {self.digest}")
                    return True
                else:
                    logger.debug(f"Nu exista model in calea {config_path} si {weight_path}")
                    return False

    def salveaza(self, config_path, weight_path):
        with self.session.as_default():
            with self.session.graph.as_default():
                set_session(self.session)
                logger.debug(f"Salveaza modelul in {config_path}")
                with open(config_path, "wt") as f:
                    json.dump(self.model.get_config(), f)
                    self.model.save_weights(weight_path)
                self.digest = self.fetch_digest(weight_path)
                logger.debug(f"Am salvat modelul folosind codul {self.digest}")
