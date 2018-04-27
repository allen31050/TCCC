CLEAN = False
W2V = "fasttext" # "glove", "fasttext"
TR_TS = "_fr" # "": original, "_es": spanish, "_fr": french, "_de": german, "_prep": preprocessed

INPUT_DIR = "../data/"
TRAIN_FILE = INPUT_DIR + "train%s.csv" %(TR_TS + ("_rearg" if TR_TS == "_prep" else ""))
TEST_FILE = INPUT_DIR + "test%s.csv" %("_prep_rearg" if TR_TS == "_prep" else "")
W2V_FILE = INPUT_DIR + {"glove": "glove.840B.300d.txt", "fasttext": "crawl-300d-2M.vec"}[W2V]

TOP_WORDS = 130000
MAX_LENGTH = 150
PATIENCE = 3
Y_CALSSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
BATCH_SIZE = 32#256
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
if os.path.exists(EMBEDDING_MATRIX) and os.path.exists(TR_X) and os.path.exists(TR_Y) and os.path.exists(TS_X):
    embedding_matrix = np.load(EMBEDDING_MATRIX)
    trX = np.load(TR_X)
    trY = np.load(TR_Y)
    tsX = np.load(TS_X)
else:
    test = pd.read_csv(TEST_FILE)

    trX = txtClean(train)
    # train["comment_text"].fillna("fillna")
    # test["comment_text"].fillna("fillna")
    # trX = train["comment_text"].str.lower()
    trY = train[Y_CALSSES].values
    tsX = txtClean(test)
    # tsX = test["comment_text"].str.lower()

    tokenizer = text.Tokenizer(num_words = TOP_WORDS, lower = True)
    tokenizer.fit_on_texts(list(trX) + list(tsX))
    trX = tokenizer.texts_to_sequences(trX)
    tsX = tokenizer.texts_to_sequences(tsX)
    trX = sequence.pad_sequences(trX, maxlen = MAX_LENGTH)
    tsX = sequence.pad_sequences(tsX, maxlen = MAX_LENGTH)

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

    # pool = Pool(cpu_count())
    # with open(W2V_FILE, encoding = "utf8") as gloveFile:
    #     embeddingDict = dict(pool.map(getCoefs, gloveFile, cpu_count()))

    # all_embs = np.stack(embeddingDict.values())
    # emb_mean, emb_std = all_embs.mean(), all_embs.std()

    # word_index = tokenizer.word_index

    # numWords = min(TOP_WORDS, len(word_index))
    # embedding_matrix = np.random.normal(emb_mean, emb_std, (numWords, EMBEDDING_SIZE))
    # for word, i in word_index.items():
    #     if i >= TOP_WORDS: continue
    #     embedding_vector = embeddingDict.get(word)
    #     if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    np.save(TR_X, trX)
    np.save(TR_Y, trY)
    np.save(TS_X, tsX)
    # np.save(EMBEDDING_MATRIX, embedding_matrix)

def get_model():
    inp = Input(shape = (MAX_LENGTH, ))
    x = Embedding(TOP_WORDS, EMBEDDING_SIZE, weights = [embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.2)(x)

    rnnOut = Bidirectional(GRU(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1))(x)

    convOut = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(rnnOut)
    avgPool = GlobalAveragePooling1D()(convOut)
    maxPool = GlobalMaxPooling1D()(convOut)    
    pooled = concatenate([avgPool, maxPool])

    x = Dropout(0.1)(pooled)
    x = Dense(len(Y_CALSSES), activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return model

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
            if current > globalBest: # self.best: #save model
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

model = get_model()
model.save_weights('init.h5')

globalBest = 0
if OOF:
    predTrY = np.zeros((train.shape[0], len(Y_CALSSES)))
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

        predTrY[valIdx] = predValY

    oof = train[['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    oof[Y_CALSSES] = predTrY
    oof.to_csv("oof.csv", index = False)
    model.load_weights('init.h5')

if TEST:
    trainX, valX, trainY, valY = train_test_split(trX, trY, train_size = 1 - VALIDATION_SPLIT)

    # model.fit(trainX, trainY, batch_size = BATCH_SIZE * (2 ** e), epochs = EPOCHS, validation_data = (valX, valY), callbacks = getCallbacks(valX, valY), verbose = 2)
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
    submission.to_csv("sub_gruConv%s_%s_topWords%s_maxLen%s_%s.csv" % (TR_TS, ("cleaned" if CLEAN else "uncleaned"), TOP_WORDS, MAX_LENGTH, W2V), index = False)
