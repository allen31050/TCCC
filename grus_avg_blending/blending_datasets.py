import pandas as pd
import numpy as np
import os
import re

models = []
for filename in os.listdir("./"):
	if filename.endswith(".csv"):
		print filename
		models.append(pd.read_csv(filename))


"""
For diversity, many prediction (on 4 settings) are blended
1. models: gru_conv, multi_gru_conv, capsule_gru_conv
2. training data: train.csv, train.prep_rearg.csv, train_de, train-es, train_fr
3. clean or not clean the training data
4. pretrained word2vec: glove, fasttext

list:
sub_gruConv_uncleaned_topWords130000_maxLen150_glove_2.csv
sub_capsuleGruConv_prep_uncleaned_topWords130000_maxLen150_fasttext.csv
sub_gruConv_es_uncleaned_topWords130000_maxLen150_fasttext.csv
sub_gruConv_prep_uncleaned_topWords130000_maxLen150_fasttext.csv
sub_gruConv_uncleaned_topWords130000_maxLen150_glove_3.csv
sub_multiGruConv_uncleaned_topWords150000_maxLen200_glove_4.csv
sub_gruConv_uncleaned_topWords130000_maxLen150_glove_1.csv
sub_capsuleGruConv_es_uncleaned_topWords130000_maxLen150_fasttext.csv
sub_gruConv_prep_uncleaned_topWords130000_maxLen150_glove.csv
blend_3gruConvs_onFullDatasets.csv
sub_gruConv_uncleaned_topWords130000_maxLen150_glove_4.csv
sub_capsuleGruConv_uncleaned_topWords130000_maxLen150_fasttext.csv
sub_multiGruConv_uncleaned_topWords150000_maxLen200_glove_1.csv
sub_capsuleGruConv_prep_uncleaned_topWords130000_maxLen150_glove.csv
sub_multiGruConv_cleaned_topWords150000_maxLen200_glove.csv
sub_capsuleGruConv_fr_uncleaned_topWords130000_maxLen150_glove.csv
sub_gruConv_de_uncleaned_topWords130000_maxLen150_fasttext.csv
sub_multiGruConv_uncleaned_topWords150000_maxLen200_glove_2.csv
sub_capsuleGruConv_de_uncleaned_topWords130000_maxLen150_fasttext.csv
sub_capsuleGruConv_es_cleaned_topWords130000_maxLen150_fasttext.csv
sub_multiGruConv_uncleaned_topWords150000_maxLen200_fasttext_1.csv
sub_capsuleGruConv_uncleaned_topWords130000_maxLen150_glove.csv
sub_multiGruConv_uncleaned_topWords150000_maxLen200_glove_3.csv
sub_capsuleGruConv_de_cleaned_topWords130000_maxLen150_glove.csv
sub_multiGruConv_uncleaned_topWords150000_maxLen200_glove.csv
sub_multiGruConv_fr_uncleaned_topWords150000_maxLen200_fasttext.csv
sub_multiGruConv_es_uncleaned_topWords150000_maxLen200_glove.csv
sub_capsuleGruConv_de_uncleaned_topWords130000_maxLen150_glove.csv
sub_multiGruConv_cleaned_topWords150000_maxLen200_fasttext_1.csv
sub_gruConv_de_uncleaned_topWords130000_maxLen150_glove.csv
sub_capsuleGruConv_fr_uncleaned_topWords130000_maxLen150_fasttext.csv
sub_gruConv_uncleaned_topWords130000_maxLen150_glove.csv
sub_gruConv_cleaned_topWords130000_maxLen150_fasttext_1.csv
sub_gruConv_es_cleaned_topWords130000_maxLen150_fasttext.csv
sub_capsuleGruConv_es_uncleaned_topWords130000_maxLen150_glove.csv
sub_capsuleGruConv_cleaned_topWords130000_maxLen150_fasttext.csv
sub_multiGruConv_cleaned_topWords150000_maxLen200_fasttext.csv
sub_multiGruConv_uncleaned_topWords150000_maxLen200_fasttext.csv
sub_capsuleGruConv_es_cleaned_topWords130000_maxLen150_glove.csv
sub_gruConv_fr_uncleaned_topWords130000_maxLen150_fasttext.csv
sub_multiGruConv_de_uncleaned_topWords150000_maxLen200_glove.csv
sub_capsuleGruConv_uncleaned_topWords130000_maxLen150_fasttext_2.csv
sub_capsuleGruConv_uncleaned_topWords130000_maxLen150_fasttext_3.csv
sub_gruConv_uncleaned_topWords130000_maxLen150_fasttext.csv
sub_capsuleGruConv_uncleaned_topWords130000_maxLen150_fasttext_1.csv
sub_multiGruConv_es_uncleaned_topWords150000_maxLen200_fasttext.csv
sub_gruConv_fr_uncleaned_topWords130000_maxLen150_glove.csv
sub_gruConv_cleaned_topWords130000_maxLen150_fasttext.csv
sub_multiGruConv_fr_uncleaned_topWords150000_maxLen200_glove.csv
sub_gruConv_uncleaned_topWords130000_maxLen150_fasttext_1.csv
sub_gruConv_es_cleaned_topWords130000_maxLen150_glove.csv
sub_capsuleGruConv_uncleaned_topWords130000_maxLen150_glove_1.csv
sub_capsuleGruConv_cleaned_topWords130000_maxLen150_glove.csv
sub_capsuleGruConv_uncleaned_topWords130000_maxLen150_glove_2.csv
"""
blend = models[0].copy()

for label in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
	ttlProb = 0
	for m in range(len(models)):
		ttlProb += models[m][label]
	blend[label] = ttlProb / len(models)

blend.to_csv('blend_3gruConvs_onFullDatasets.csv', index = False)


