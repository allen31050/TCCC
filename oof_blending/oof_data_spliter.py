INPUT_DIR = "../data/"
OUTPUT_DIR = INPUT_DIR + "data_for_oof_training/"
Y_CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

train = pd.read_csv(INPUT_DIR + "train.csv")

kf = KFold(n_splits = 5, shuffle = True)

dummy = range(len(train))
for n, (trainIdx, valIdx) in enumerate(kf.split(train, dummy)):
    trainData, valData = train.iloc[trainIdx], train.iloc[valIdx]
    trainData.to_csv(OUTPUT_DIR + "trainOOF" + str(n + 1) + ".csv", index = False)
    
    samSub = valData
    
    valData = valData.drop(columns = Y_CLASSES)
    valData.to_csv(OUTPUT_DIR + "valOOF" + str(n + 1) + ".csv", index = False)
    
    samSub = samSub.drop(columns=["comment_text"])
    samSub[Y_CLASSES] = 0.5
    samSub.to_csv(OUTPUT_DIR + "sample_submissionOOF" + str(n + 1) + ".csv", index = False)