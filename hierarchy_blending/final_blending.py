import pandas as pd
import numpy as np

models = [
pd.read_csv("blend_otherModels.csv"), #0.9862
pd.read_csv("blend_3gruConvs_onFullDatasets.csv"), #0.9861
pd.read_csv('blend_it_all.csv'), #0.9868
# pd.read_csv('lazy_ensemble_submission.csv'), #0.9865
]

blend = models[0].copy()

for label in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
	ttlProb = 0
	for m in range(len(models)):
		ttlProb += models[m][label]
	blend[label] = ttlProb / len(models)

blend.to_csv('hierarchy_gigaBlend_fitMore.csv', index = False)


