CLEAN = True
W2V = "glove" # "glove", "fasttext"
TR_TS = "_de" # "": original, "_es": spanish, "_fr": french, "_de": german, "_prep": preprocessed

INPUT_DIR = "../data/"
TRAIN_FILE = INPUT_DIR + "train%s.csv" %(TR_TS + ("_rearg" if TR_TS == "_prep" else ""))
TEST_FILE = INPUT_DIR + "test%s.csv" %("_prep_rearg" if TR_TS == "_prep" else "")
W2V_FILE = INPUT_DIR + {"glove": "glove.840B.300d.txt", "fasttext": "crawl-300d-2M.vec"}[W2V]

TOP_WORDS = 130000
MAX_LENGTH = 150
PATIENCE = 3
Y_CALSSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
BATCH_SIZE = 256
EPOCHS = 40
EMBEDDING_SIZE = 300
NUM_CPU = 4
VALIDATION_SPLIT = 0.1
FOLDS = 5
OOF = False
TEST = True

CKPT_PATH = "weights.best.hdf5"
EMBEDDING_MATRIX = INPUT_DIR + "embeddingMatrix%s_topWords%s_%s.npy" % (TR_TS, TOP_WORDS, W2V)
TR_X = INPUT_DIR + "trX%s_%s_topWords%s_maxLen%s_%s.npy" % (TR_TS, ("cleaned" if CLEAN else "uncleaned"), TOP_WORDS, MAX_LENGTH, W2V)
TR_Y = INPUT_DIR + "trY.npy"
TS_X = INPUT_DIR + "tsX%s_%s_topWords%s_maxLen%s_%s.npy" % (TR_TS, ("cleaned" if CLEAN else "uncleaned"), TOP_WORDS, MAX_LENGTH, W2V)

import os
import re
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from keras.models import *
from keras.layers.core import *
from keras.layers import *
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from subprocess import check_output
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

def txtClean(df):
    if CLEAN: df["comment_text"] = [" ".join([((word + "!") if word.isupper() and word != "I" else word.lower()) for word in comment.split()]) for comment in df["comment_text"]]
    df["comment_text"] = df["comment_text"].str.lower()
    if CLEAN:
        df["comment_text"] = df["comment_text"].str.replace("won\'t", "will not")
        df["comment_text"] = df["comment_text"].str.replace("can\'t", "can not")
        df["comment_text"] = df["comment_text"].str.replace("\'m", " am")
        df["comment_text"] = df["comment_text"].str.replace("\'ll", " will")
        df["comment_text"] = df["comment_text"].str.replace("n\'t", " not")
        df["comment_text"] = df["comment_text"].str.replace("\'ve", " have")
        df["comment_text"] = df["comment_text"].str.replace("\'s", " is")
        df["comment_text"] = df["comment_text"].str.replace("\'re", " are")
        df["comment_text"] = df["comment_text"].str.replace("\'d", " would")
        df["comment_text"] = df["comment_text"].str.replace("!", " ! ")
        df["comment_text"] = df["comment_text"].str.replace("?", " ? ")
        df["comment_text"] = df["comment_text"].str.replace("\\.\\.+", " ... ")
        df["comment_text"] = df["comment_text"].str.replace(":", " : ")
        df["comment_text"] = df["comment_text"].str.replace(",", " , ")
        df["comment_text"] = df["comment_text"].str.replace("\"", " \" ")
    list_sentences = df["comment_text"].fillna("lilili").values
    return list_sentences

train = pd.read_csv(TRAIN_FILE)
train = train.sample(frac = 1)
# train["comment_text"].fillna("fillna")
if os.path.exists(EMBEDDING_MATRIX) and os.path.exists(TR_X) and os.path.exists(TR_Y) and os.path.exists(TS_X):
    embedding_matrix = np.load(EMBEDDING_MATRIX)
    trX = np.load(TR_X)
    trY = np.load(TR_Y)
    tsX = np.load(TS_X)
else:
    test = pd.read_csv(TEST_FILE)

    trX = txtClean(train)
    trY = train[Y_CALSSES].values
    tsX = txtClean(test)

    tokenizer = text.Tokenizer(num_words = TOP_WORDS, lower = True)
    tokenizer.fit_on_texts(list(trX) + list(tsX))
    trX = tokenizer.texts_to_sequences(trX)
    tsX = tokenizer.texts_to_sequences(tsX)
    trX = sequence.pad_sequences(trX, maxlen = MAX_LENGTH)
    tsX = sequence.pad_sequences(tsX, maxlen = MAX_LENGTH)

    # test = pd.read_csv(INPUT_DIR + "test.csv")
    # test["comment_text"].fillna("fillna")

    # tokenizer = text.Tokenizer(num_words = TOP_WORDS, lower = True)
    # list_sentences_train = train["comment_text"].str.lower()
    # list_sentences_test = test["comment_text"].str.lower()
    # tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
    
    # list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    # trX = sequence.pad_sequences(list_tokenized_train, maxlen = MAX_LENGTH)
    # list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    # tsX = sequence.pad_sequences(list_tokenized_test, maxlen = MAX_LENGTH)

    # trY = train[Y_CALSSES].values

    def getCoefs(row):
        row = row.strip().split()
        word, arr = " ".join(row[ : -EMBEDDING_SIZE]), row[-EMBEDDING_SIZE : ]
        return word, np.asarray(arr, dtype = "float32")

    if os.path.exists(EMBEDDING_MATRIX):
        embedding_matrix = np.load(EMBEDDING_MATRIX)
    else:
        pool = Pool(cpu_count())
        with open(W2V_FILE, encoding = "utf8") as pretrainedFile:
            embeddingDict = dict(pool.map(getCoefs, pretrainedFile, cpu_count()))
        all_embs = np.stack(embeddingDict.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        word_index = tokenizer.word_index
        numWords = min(TOP_WORDS, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (numWords, EMBEDDING_SIZE))
        for word, i in word_index.items():
            if i >= TOP_WORDS: continue
            embedding_vector = embeddingDict.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        np.save(EMBEDDING_MATRIX, embedding_matrix)

    np.save(TR_X, trX)
    np.save(TR_Y, trY)
    np.save(TS_X, tsX)
    

def squash(x, axis=-1):
    # s_squared_norm is really small
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    # if (abs(s_squared_norm) < K.epsilon()) s_squared_norm = np.sign(s_squared_norm) * K.epsilon()
    # scale = K.sqrt(s_squared_norm + K.epsilon())
    # return x / scale

# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def get_model():
    inp = Input(shape = (MAX_LENGTH, ))
    x = Embedding(TOP_WORDS, EMBEDDING_SIZE, weights = [embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.28)(x)
    
    rnnOut = Bidirectional(GRU(128, activation='relu', return_sequences = True, dropout = 0.25, recurrent_dropout = 0.25))(x)

    capsule = Capsule(num_capsule = 10, dim_capsule = 16, routings = 5, share_weights = True)(rnnOut)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)

    x = Dropout(0.28)(capsule)
    x = Dense(len(Y_CALSSES), activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return model

model = get_model()
model.save_weights('init.h5')

class RocAucEvaluation(Callback):
    def __init__(self, filepath = CKPT_PATH, validation_data=(), interval=10, max_epoch = 20):
        super(Callback, self).__init__()

        self.interval = interval
        self.filepath = filepath
        self.stopped_epoch = max_epoch
        # self.best = 0
        self.X_val, self.y_val = validation_data
        self.y_pred = np.zeros(self.y_val.shape)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            
            # Important lines
            current = roc_auc_score(self.y_val, y_pred)
            logs['roc_auc_val'] = current

            print(" - AUC - score: {:.5f}".format(current))
            
            global globalBest
            if current > globalBest: #save model
            # if current > self.best: #save model
                # self.best = current
                globalBest = current
                self.y_pred = y_pred
                self.stopped_epoch = epoch+1
                self.model.save(self.filepath, overwrite = True)
                print("AUC score improved, model saved\n")
            else:
                print("AUC score not improved\n")


def getCallbacks(valX, valY):
    RocAucVal = RocAucEvaluation(validation_data = (valX, valY), interval = 1)
    # earlyStop = EarlyStopping(monitor = 'roc_auc_val', patience = PATIENCE, mode = 'max', verbose = 1)
    # return [RocAucVal, earlyStop]
    return [RocAucVal]

globalBest = 0
if OOF:
    oofPredTrY = np.zeros((train.shape[0], len(Y_CALSSES)))
    kf = KFold(n_splits = FOLDS, shuffle = True)

    dummy = range(len(trX))
    for trainIdx, valIdx in kf.split(trX, dummy):
        trainX, valX = trX[trainIdx], trX[valIdx]
        trainY, valY = trY[trainIdx], trY[valIdx]

        model.load_weights('init.h5')
        
        # model.fit(trainX, trainY, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (valX, valY), callbacks = getCallbacks(valX, valY), verbose = 2, shuffle = False)
        globalBest = 0
        countEarlyStop = 0
        for e in range(EPOCHS):
            lastgloBest = globalBest
            model.fit(trainX, trainY, batch_size = min(1024, BATCH_SIZE * (2 ** e)), epochs = 1, validation_data = (valX, valY), callbacks = getCallbacks(valX, valY), verbose = 2, shuffle = False)
            if (globalBest == lastgloBest): countEarlyStop += 1
            if countEarlyStop >= PATIENCE:
                print("Early stopping...")
                break

        model.load_weights(CKPT_PATH)
        predValY = model.predict(valX, batch_size = 1024, verbose = 2)
        
        oofPredTrY[valIdx] = predValY

    oof = train[['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    oof[Y_CALSSES] = oofPredTrY
    oof.to_csv("oof.csv", index = False)
    model.load_weights('init.h5')

if TEST:
    trainX, valX, trainY, valY = train_test_split(trX, trY, train_size = 1 - VALIDATION_SPLIT)

    # model.fit(trainX, trainY, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (valX, valY), callbacks = getCallbacks(valX, valY), verbose = 2)
    globalBest = 0
    countEarlyStop = 0
    for e in range(EPOCHS):
        lastgloBest = globalBest
        model.fit(trainX, trainY, batch_size = min(1024, BATCH_SIZE * (2 ** e)), epochs = 1, validation_data = (valX, valY), callbacks = getCallbacks(valX, valY), verbose = 2)
        if (globalBest == lastgloBest): countEarlyStop += 1
        if countEarlyStop >= PATIENCE:
            print("Early stopping...")
            break

    model.load_weights(CKPT_PATH)
    tsY = model.predict(tsX, batch_size = 1024, verbose = 2)

    submission = pd.read_csv(INPUT_DIR + "sample_submission.csv")
    submission[Y_CALSSES] = tsY
    submission.to_csv("sub_capsuleGruConv%s_%s_topWords%s_maxLen%s_%s.csv" % (TR_TS, ("cleaned" if CLEAN else "uncleaned"), TOP_WORDS, MAX_LENGTH, W2V), index = False)
