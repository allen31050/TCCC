import pandas as pd
import numpy as np
import os
import re

models = []
for filename in os.listdir("./"):
	if filename.endswith(".csv"):
		print filename
		models.append(pd.read_csv(filename))


blend = models[0].copy()

for label in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
	ttlProb = 0
	for m in range(len(models)):
		ttlProb += models[m][label]
	blend[label] = ttlProb / len(models)

blend.to_csv('sub21_blend.csv', index = False)


