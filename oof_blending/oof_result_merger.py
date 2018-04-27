INPUT_DIR = "../data/oof_results_split/"
MODEL_NUM = 13
Y_LCASS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
import pandas as pd
for MODEL_NUM in [20]:
	oof = pd.read_csv("../data/train.csv")
	oof = oof.drop(columns=["comment_text"] + Y_LCASS)

	oof1 = pd.read_csv(INPUT_DIR + "oof%s_1.csv" %MODEL_NUM)
	oof2 = pd.read_csv(INPUT_DIR + "oof%s_2.csv" %MODEL_NUM)
	oof3 = pd.read_csv(INPUT_DIR + "oof%s_3.csv" %MODEL_NUM)
	oof4 = pd.read_csv(INPUT_DIR + "oof%s_4.csv" %MODEL_NUM)
	oof5 = pd.read_csv(INPUT_DIR + "oof%s_5.csv" %MODEL_NUM)

	oof1to5 = pd.concat([oof1, oof2, oof3, oof4, oof5])
	# oof1to5 = oof1to5.drop(columns=["comment_text"]) # fix those oof with comment_text

	oof = oof.join(oof1to5.set_index('id'), on='id')
	oof.to_csv("oof" + str(MODEL_NUM) + ".csv", index = False)
