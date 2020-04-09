import numpy as np
# from sklearn import svm
# from sklearn import tree
import pyreadstat
from category_encoders import JamesSteinEncoder, MEstimateEncoder, HelmertEncoder, BackwardDifferenceEncoder, WOEEncoder, PolynomialEncoder
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from keras.metrics import AUC, Recall
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn import metrics
import keras.backend as K
from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout, GRU, Conv1D, Flatten, Permute, Subtract, Add, Dot, Concatenate, BatchNormalization, Embedding
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd


class DeepNN:

    # Initialization
    def __init__(self, nb_classes,
                 seed=False,
                 proposed_mode=False):
        if seed:
            np.random.seed(0)
        self.nb_classes = nb_classes
        self.model = None
        self.proposed_mode = proposed_mode

    # Load model
    def load_model(self, filepath):
        self.model = load_model(filepath)

    # Configuration
    def config(self, layers):
        # Input layer
        input_layer = Input(shape=(layers[0], 1))
        # Dropout

        # Hidden Layer 1
        encoded = Dense(layers[1],
                        activation='relu')(input_layer)
        # Dropout
        # Hidden Layer 2
        encoded = Conv1D(layers[2], 5,
                         activation='relu')(encoded)
        # Dropout
        encoded = Dropout(0.5)(encoded)
        encoded = Flatten()(encoded)
        # Hidden Layer 3
        encoded = Dense(layers[3],
                        activation='relu')(encoded)
        # Dropout
        # encoded = Dropout(0.5)(encoded)
        # Softmax
        softmax = Dense(self.nb_classes,
                        activation='softmax')(encoded)
        # Config the model
        self.model = Model(
            input=input_layer,
            output=softmax)
        # autoencoder compilation
        self.model.compile(optimizer='nadam',
                           loss="categorical_crossentropy",
                           metrics=[Recall(), AUC()])

    # Fit
    def fit(self, X_train, y_train, \
            batch_size=128, nb_epoch=20,
            validation_split=0.0,
            modelpath='weights.hdf5',
            shuffle=False):

        # a list of callbacks
        callbacks = []

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, self.nb_classes)

        # proposed checkpoint
        checkpointer = ModelCheckpoint(
            filepath=modelpath,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min')

        if self.proposed_mode:
            validation_split = 0.2
            callbacks.append(checkpointer)

        ratio = np.bincount(y_train.astype(int))
        ratio = float(ratio[0]) / ratio[1]
        ratio = {1: 5, 0: ratio}
        # print ratio

        history = self.model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            verbose=1,
            validation_split=validation_split,
            # This turns the Deep NN into cost-sensitive mode
            class_weight=ratio,
            callbacks=callbacks)

        if self.proposed_mode:
            self.load_model(modelpath)

    def _proba(self, X_test):
        return self.model.predict(X_test)

    def _predict(self, X_test):
        proba = self.predict_proba(X_test)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X_test):
        return self._proba(X_test)

    def predict(self, X_test):
        return self._predict(X_test)

    def evaluate(self, X_test, y_test):
        Y_test = np_utils.to_categorical(y_test, self.nb_classes)
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        # specificity = specificity_score(y_test, y_pred)
        # gmean = np.sqrt(recall * specificity)
        f1 = metrics.f1_score(y_test, y_pred)
        pr_auc = metrics.average_precision_score(Y_test, y_proba)
        roc_auc = metrics.roc_auc_score(Y_test, y_proba)
        return confusion_matrix, precision, recall, f1, pr_auc, roc_auc


def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_classif, k=25)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


k_latent = 2
embedding_reg = 0.0002
kernel_reg = 0.1


def get_embed(x_input, x_size, k_latent):
    if x_size > 0:  # category
        embed = Embedding(x_size, k_latent, input_length=1,
                          embeddings_regularizer=l2(embedding_reg))(x_input)
        embed = Flatten()(embed)
    else:
        embed = Dense(k_latent, kernel_regularizer=l2(embedding_reg))(x_input)
    return embed


def build_model_1(X, f_size):
    dim_input = len(f_size)

    input_x = [Input(shape=(1,)) for i in range(dim_input)]

    biases = [get_embed(x, size, 1) for (x, size) in zip(input_x, f_size)]

    factors = [get_embed(x, size, k_latent) for (x, size) in zip(input_x, f_size)]

    s = Add()(factors)

    diffs = [Subtract()([s, x]) for x in factors]

    dots = [Dot(axes=1)([d, x]) for d, x in zip(diffs, factors)]

    x = Concatenate()(biases + dots)
    x = BatchNormalization()(x)
    output = Dense(1, activation='relu', kernel_regularizer=l2(kernel_reg))(x)
    model = Model(inputs=input_x, outputs=[output])
    opt = Adam(clipnorm=0.5)
    model.compile(optimizer=opt, loss='mean_squared_error')
    output_f = factors + biases
    model_features = Model(inputs=input_x, outputs=output_f)
    return model, model_features


if __name__ == '__main__':
    # this function prints the metrics in CSV format
    def show(score):
        confusion_matrix, precision, recall, f1, prc_auc, roc_auc = score
        print(
            "TN,FP,FN,TP,Precision,Recall,F1,PRC,ROC")
        print(
            "%d,%d,%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f" \
            % (confusion_matrix[0, 0], confusion_matrix[0, 1],
               confusion_matrix[1, 0], confusion_matrix[1, 1],
               precision, recall, f1, prc_auc, roc_auc))


    df, meta = pyreadstat.read_sav("CMS.sav")
    cols = df.columns
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df = df.select_dtypes(include=['float32', 'float64', 'int'])
    imp.fit(df.values)
    df = imp.transform(df.values)

    CMS = np.round(df[:, -1])
    # df = df.select_dtypes(include=['float32', 'float64', 'int'])
    X = df[:, 0:df.shape[1] - 1:1]
    CMS = CMS - 1
    CMS[CMS == -1] = 1

    Xd = pd.DataFrame(X)
    duplicatedItem = Xd.duplicated(keep='first')
    X = X[duplicatedItem == False, :]
    CMS = CMS[duplicatedItem == False]
    # X = np.append(X, np.reshape(np.sum(X, axis=1), (X.shape[0], 1)), axis=1)

    n_epochs = 100
    P = 17

    batch_size = 2 ** P
    print(batch_size)
    f_size = [int(X[:, f].max()) + 1 for f in range(X.shape[1])]
    X = [X[:, f] for f in range(X.shape[1])]
    model, model_features = build_model_1(X, f_size)
    earlystopper = EarlyStopping(patience=0, verbose=1)
    w_train = (30 * (CMS == 1).astype('float32') + 1).ravel()
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    model.fit(X, CMS,
              epochs=n_epochs, batch_size=batch_size, verbose=1, shuffle=True,
              sample_weight=w_train,
              callbacks=[earlystopper])
    X_ = model_features.predict(X, batch_size=batch_size)


    X = X_[0]
    for l in range(1, len(X_)):
        X = np.append(X, X_[l], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, CMS, test_size=0.2, stratify=CMS)

    # X_train, X_test, _ = select_features(X_train, y_train, X_test)
    # ros = RandomOverSampler()
    # smt = SMOTETomek()
    # X_train, y_train = smt.fit_sample(X_train, y_train)
    # obtain the number of classes
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    nb_classes = np.size(np.unique(y_train))

    # training
    # proposed_mode = True, using the validation-loss strategy
    # proposed_mode = False, using the normal strategy
    deepNN = DeepNN(nb_classes, seed=True, proposed_mode=False)
    deepNN.config(layers=[X_train.shape[1], 128, 32, 64])
    deepNN.fit(X_train, y_train, nb_epoch=2, validation_split=.1, shuffle=True)

    # evaluate on training data

    score = deepNN.evaluate(X_train, y_train)
    show(score)


    score = deepNN.evaluate(X_test, y_test)
    show(score)

    model.save("model.h5")
    model_features.save("model_feature.h5")
    deepNN.model.save("classificationModel.h5")
