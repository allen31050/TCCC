Y_CLASS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
import pandas as pd
# df = pd.read_csv("train_prep.csv")
# df = df.drop(columns=["set"])
# cols = df.columns.tolist()
# cols = [cols[1], cols[0], cols[7], cols[5], cols[4], cols[6], cols[3], cols[2]]
# df = df[cols]
# df.to_csv("train_prep_rearg.csv", index = False)

df = pd.read_csv("test_prep.csv")
df = df.drop(columns=["set"] + Y_CLASS)
cols = df.columns.tolist()
cols = [cols[1], cols[0]]
df = df[cols]
df.to_csv("test_prep_rearg.csv", index = False)
