import settings
import keras
from keras.layers import Input,  Dropout,   Conv2D, MaxPooling2D,  Conv2DTranspose, concatenate
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import Accuracy
from keras.callbacks import ModelCheckpoint, EarlyStopping


class unet(object):

    def __init__(self, fms=settings.FEATURE_MAPS,
                 output_path=settings.OUT_PATH,
                 inference_filename=settings.INFERENCE_FILENAME,
                 learning_rate=settings.LEARNING_RATE,
                 use_dropout=settings.USE_DROPOUT):

        self.fms = fms  # 32 or 16 depending on your memory size

        self.learningrate = learning_rate

        self.output_path = output_path
        self.inference_filename = inference_filename

        self.optimizer = Adam(learning_rate=self.learningrate)
        self.loss = BinaryCrossentropy()
        self.metrics = [Accuracy()]
        self.concat_axis = 3

        self.use_dropout = use_dropout

    def unet_model(self, imgs_shape, final=False):
        """
        U-Net Model
        ===========
        Based on https://arxiv.org/abs/1505.04597
        """

        self.input_shape = imgs_shape

        inputs = Input(self.input_shape, name="Electron Microscopy")

        # Convolution parameters
        params = dict(kernel_size=(3, 3), activation="relu",
                      padding="same",
                      kernel_initializer="he_uniform")

        # Transposed convolution parameters
        params_trans = dict(kernel_size=(3, 3), strides=(2, 2), padding="same")

        """---------------------------CONTRACTION------------------------------------------"""
        encodeA = Conv2D(
            name="encodeAa", filters=self.fms, **params)(inputs)
        if self.use_dropout:
            encodeA = Dropout(rate=settings.DROPOUT_RATE)(encodeA)
        encodeA = Conv2D(
            name="encodeAb", filters=self.fms, **params)(encodeA)
        poolA = MaxPooling2D(name="poolA", pool_size=(2, 2))(encodeA)

        encodeB = Conv2D(
            name="encodeBa", filters=self.fms*2, **params)(poolA)
        if self.use_dropout:
            encodeB = Dropout(rate=settings.DROPOUT_RATE)(encodeB)
        encodeB = Conv2D(
            name="encodeBb", filters=self.fms*2, **params)(encodeB)
        poolB = MaxPooling2D(name="poolB", pool_size=(2, 2))(encodeB)

        encodeC = Conv2D(
            name="encodeCa", filters=self.fms*4, **params)(poolB)
        if self.use_dropout:
            encodeC = Dropout(rate=settings.DROPOUT_RATE)(encodeC)
        encodeC = Conv2D(
            name="encodeCb", filters=self.fms*4, **params)(encodeC)
        poolC = MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeC)

        encodeD = Conv2D(
            name="encodeDa", filters=self.fms*8, **params)(poolC)
        if self.use_dropout:
            encodeD = Dropout(rate=settings.DROPOUT_RATE)(encodeD)
        encodeD = Conv2D(
            name="encodeDb", filters=self.fms*8, **params)(encodeD)
        poolD = MaxPooling2D(name="poolD", pool_size=(2, 2))(encodeD)

        encodeE = Conv2D(
            name="encodeEa", filters=self.fms*16, **params)(poolD)
        if self.use_dropout:
            encodeE = Dropout(rate=settings.DROPOUT_RATE)(encodeE)
        encodeE = Conv2D(
            name="encodeEb", filters=self.fms*16, **params)(encodeE)

        """---------------------------EXPANSION------------------------------------------"""

        up = Conv2DTranspose(name="transconvE", filters=self.fms*8,
                             **params_trans)(encodeE)
        concatD = concatenate(
            [up, encodeD], axis=self.concat_axis, name="concatD")
        decodeC = Conv2D(
            name="decodeCa", filters=self.fms*8, **params)(concatD)
        if self.use_dropout:
            decodeC = Dropout(rate=settings.DROPOUT_RATE)(decodeC)
        decodeC = Conv2D(
            name="decodeCb", filters=self.fms*8, **params)(decodeC)

        up = Conv2DTranspose(name="transconvC", filters=self.fms*4,
                             **params_trans)(decodeC)
        concatC = concatenate(
            [up, encodeC], axis=self.concat_axis, name="concatC")

        decodeB = Conv2D(
            name="decodeBa", filters=self.fms*4, **params)(concatC)
        if self.use_dropout:
            decodeB = Dropout(rate=settings.DROPOUT_RATE)(decodeB)
        decodeB = Conv2D(
            name="decodeBb", filters=self.fms*4, **params)(decodeB)

        up = Conv2DTranspose(name="transconvB", filters=self.fms*2,
                             **params_trans)(decodeB)
        concatB = concatenate(
            [up, encodeB], axis=self.concat_axis, name="concatB")

        decodeA = Conv2D(
            name="decodeAa", filters=self.fms*2, **params)(concatB)
        if self.use_dropout:
            decodeA = Dropout(rate=settings.DROPOUT_RATE)(decodeA)
        decodeA = Conv2D(
            name="decodeAb", filters=self.fms*2, **params)(decodeA)

        up = Conv2DTranspose(name="transconvA", filters=self.fms,
                             **params_trans)(decodeA)
        concatA = concatenate(
            [up, encodeA], axis=self.concat_axis, name="concatA")

        convOut = Conv2D(
            name="convOuta", filters=self.fms, **params)(concatA)
        if self.use_dropout:
            convOut = Dropout(rate=settings.DROPOUT_RATE)(convOut)
        convOut = Conv2D(
            name="convOutb", filters=self.fms, **params)(convOut)

        prediction = Conv2D(name="PredictionMask",
                            filters=settings.NUM_CHANNELS_OUT, kernel_size=(
                                1, 1),
                            activation="sigmoid")(convOut)

        model = keras.Model(inputs=[inputs], outputs=[
            prediction], name="UNet_Electron_Microscopy")

        optimizer = self.optimizer

        if final:
            model.trainable = False
        else:

            model.compile(optimizer=optimizer,
                          loss=self.loss,
                          metrics=self.metrics)

            if self.print_model:
                model.summary()

        return model

    def get_callbacks(self):
        """
        Define any callbacks for the training
        """

        model_filename = os.path.join(
            self.output_path, self.inference_filename)

        print("Writing model to '{}'".format(model_filename))

        # Save model whenever we get better validation loss
        model_checkpoint = ModelCheckpoint(model_filename,
                                           verbose=1,
                                           monitor="val_loss",
                                           save_best_only=True)

        early_stopping = EarlyStopping(
            patience=5, restore_best_weights=True)

        return model_filename, [model_checkpoint, early_stopping]

    def create_model(self, imgs_shape, final=False):
        return self.unet_model(imgs_shape, final=final)

    def load_model(self, model_filename):
        """
        Load a model from Keras file
        """
        return keras.models.load_model(model_filename)
