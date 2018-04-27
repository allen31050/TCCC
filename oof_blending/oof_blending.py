from os import listdir
from os.path import isfile
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings(action = "ignore", category = DeprecationWarning, module = "sklearn")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from multiprocessing import cpu_count, Pool

# next step: https://github.com/Godricly/comment_toxic

badwordSet = [
r"d[!i]c?ke?",r"[@a8][$sz]{2,}","a s s","anus","butthole","buttwipe",r"ba[zs]*t[ae]rd",r"b[i!1]a?[7+t]ch","s.o.b.",r"blow ?job",
r"c[o0]c?k",r"fu[c(]?k?",r"fc?u?k","f u c k","f you","f uck","fu ck","fcuk","f ing",r"w ?t ?f","s t f u","jerk",r"ji[sz]+i?m?","jackoff",
r"[ck]u?nt","cum",r"Le[sz]+bian",r"sh[iy!1][t+]",r"d[a4]mn","hell",r"h[oa0]+r[me]?","hoer","h4x0r","hoer",r"b[0o]{2,}b","breast",
r"puss[ye]+","gay","lesbian","lesbo",r"n[i1]+g+[ue]?r","nigga",r"fai?g+[1io]?t?","feg",r"p[0o]rn",r"pr[0o]n",r"idi[o0]t",
r"maso[kch]+ist",r"masst[ue]?rbai?t[3e]?","nast","orafis",r"orgas[ui]?m","knob","jap","boffing","suck","sux",
"carpet muncher",r"[kc]awk","clit","crap",r"dil+d[o0]",r"dominatri","dyke","enema","fart","fudge packer","g00k",
r"orif[ia][sce]+",r"pack[yie]+","pecker",r"pe+i?n[u1ai]+s",r"phu[kc]+",r"pola[ck]+","Poonani",r"pr[i1][ck]+","puuke",
r"q[wu]e[ei]r",r"reck?tum","retard","sadist","scank","schlong","screwing","semen","sex",r"skanc?k","skank","slut",
"tit","turd",r"va1?[gj][1i]+na",r"vul+va",r"w[o0]p","xrated","xxx","arschloch","boiolas","buceta","chink","cipa",
"dirsa",r"ejac?kulate","fux0r","jism",r"l3i[+t]ch","mofo","nazi","nutsack","phuck","pimpis","scrotum","shemale","smut",r"t[ei]+ts",
"teez","testical","testicle","titt",r"w[o0]{2,}se","wank","dyke","amcik","andskota","arse","assrammer","ayir","bollock*",
"butt-pirate","cabron","cazzo","chraa","chuj","daygo","dego","dupa","dziwka","Ekrem","Ekto","enculer","faen","fanculo","fanny","feces",
"felcher","ficken","fitt","flikker","foreskin","Fotze","futkretzn","gook","guiena","helvete","honkey","huevon","hui","injun","kanker",
"kike","klootzak","kraut","knulle","kuk","Kurac","kurwa","kusi","kyrpa","mamhoon","masturbat*","merd","mibun","monkleigh","mouliewop",
"muie","mulkku","muschi","nepesaurio","orospu","paska","perse","picka","pierdol","pillu","pimmel","piss","pizda","poontsee","poop",
"preteen",r"pul[ea]",r"put[ao]","qahbeh","queef","rautenberg","schaffer","scheiss","schlampe","schmuck","screw",r"sharmut[ae]",
"shipal","shiz","skribz","skurwysyn","sphencter","spic","spierdalaj","splooge","suka","vittu","wetback","wichser","yed","zabourah"]

#######################
# FEATURE ENGINEERING #
#######################
"""
Main function
Input: pandas Series and a feature engineering function
Output: pandas Series
"""
def engineer_feature(series, func, normalize=True):
    # feature = series.apply(func)
    pool = Pool(cpu_count())
    feature = pd.Series(pool.map(func, series.values))
    pool.close()
    pool.join()

    if normalize:
        feature = pd.Series(z_normalize(feature.values.reshape(-1,1)).reshape(-1,))
    feature.name = func.__name__ 
    return feature

"""
Engineer features
Input: pandas Series and a list of feature engineering functions
Output: pandas DataFrame
"""
def engineer_features(series, funclist, normalize=True):
    features = pd.DataFrame()
    for func in funclist:
        feature = engineer_feature(series, func, normalize)
        features[feature.name] = feature
    return features

"""
Normalizer
Input: NumPy array
Output: NumPy array
"""
scaler = StandardScaler()
def z_normalize(data):
    scaler.fit(data)
    return scaler.transform(data)


"""
Feature functions
"""
# source: https://www.kaggle.com/ogrellier/lgbm-with-words-and-chars-n-gram/code
# source: https://www.kaggle.com/allen505378/training-weak-learner-with-pr-fane-dict-lookup
def raw_word_num(x): return len(x.split())
def cap_word_freq(x): return sum([word.isupper() for word in x.split()])
def asterix_freq(x): return x.count('!')
def questionmark_freq(x): return x.count('?')
def hashtag_freq(x): return x.count('#')
def star_freq(x): return x.count('*')
def dot_freq(x): return x.count('.')
def ellipsis(x): return len(re.findall('...',x))
def uppercase_freq(x): return len(re.findall(r'[A-Z]',x))
def ant_slash_n(x): return len(re.findall(r"\n", x))
def nb_fk(x): return len(re.findall(r"[Ff]\S{2}[Kk]", x))
def nb_sk(x): return len(re.findall(r"[Ss]\S{2}[Kk]", x))
def nb_dk(x): return len(re.findall(r"[dD]ick", x))
def nb_you(x): return (len(re.findall(r"\WYOU\W", x)) + len(re.findall(r"\Wyou\W", x)))
def nb_mother(x): return len(re.findall(r"\Wmother\W", x))
def nb_ng(x): return len(re.findall(r"\Wnigger\W", x))
def start_with_columns(x): return len(re.findall(r"^\:+", x))
def has_timestamp(x): return len(re.findall(r"\d{2}|:\d{2}", x))
def has_date_long(x): return len(re.findall(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
def has_date_short(x): return len(re.findall(r"\D\d{1,2} \w+ \d{4}", x))
def has_http(x): return len(re.findall(r"http[s]{0,1}://\S+", x))
def has_mail(x): return len(re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", x))
def has_emphasize_equal(x): return len(re.findall(r"\={2}.+\={2}", x))
def has_emphasize_quotes(x): return len(re.findall(r"\"{4}\S+\"{4}", x))
def badword_freq(x): return sum([len(re.findall(bw, x.lower())) for bw in badwordSet])
def num_emoji(x): return len(re.findall(r"([:;=]-?[\DP])", x))

"""
Import submission and OOF files
"""
def get_subs(nums):
    hasBlend = [6,11,12,13,14,15,16,17,18,19,20,21]
    subs = np.hstack([np.array(pd.read_csv(("../data/trained_models/sub%s_blend.csv" % num) if num in hasBlend else ("../data/trained_models/sub%s.csv" % num))[LABELS]) for num in subnums])
    oofs = np.hstack([np.array(pd.read_csv("../data/trained_models/oof%s.csv" % num)[LABELS]) for num in subnums])
    return subs, oofs

if __name__ == "__main__":
    
    train = pd.read_csv('../data/train.csv').fillna(' ')
    test = pd.read_csv('../data/test.csv').fillna(' ')
    sub = pd.read_csv('../data/sample_submission.csv')
    INPUT_COLUMN = "comment_text"
    LABELS = train.columns[2:]
    
    # Import submissions and OOF files
    #exclude 0: gru_conv 0.9844
    # 1: https://www.kaggle.com/tilii7/tuned-logreg-oof-files/code 0.9802
    # 2: https://www.kaggle.com/shujian/stack-1-wordbatch-1-3-3-fm-ftrl-3/code 0.9813
    # 3: https://www.kaggle.com/rednivrug/5-fold-ridge-oof/code 0.9809
    # 4: https://www.kaggle.com/iamprateek/can-i-classify-toxic-comments/notebook (lgbm) 0.9792
    # 5: https://www.kaggle.com/ogrellier/lgbm-with-words-and-chars-n-gram 0.9792
    # 6: https://www.kaggle.com/tunguz/5-fold-tuned-logreg-oof-files 0.9803
    # 7: https://www.kaggle.com/tunguz/cnn-glove300-3-oof-4-epochs/code 0.9780
    # 8: https://www.kaggle.com/fizzbuzz/cnn-3-out-of-fold-4-epochs-preprocessed-data 0.9809
    #exclude running 9: multi_gru_conv 0.9836
    #exclude running 10: capsule_gru_conv oofing@12 0.9840
    # 11_blend: https://www.kaggle.com/shujian/textcnn-2d-convolution/code 0.9821 
    # 12: https://www.kaggle.com/tottenham/ridge-with-words-and-char-n-grams-lb-0-9809 0.9809
    # 13: https://www.kaggle.com/mmpossi/logitcomment-preprocessing 0.9809
    # 14: https://www.kaggle.com/chernyshov/linear-regression-with-preprocessing 0.9804
    # 15: https://www.kaggle.com/michaelsnell/conv1d-dpcnn-in-keras?scriptVersionId=2778234 0.9810
    # 16: https://www.kaggle.com/kailex/tidy-xgboost-glmnet-text2vec-lsa?scriptVersionId=2713975 0.9788
    # 17: https://www.kaggle.com/peterhurford/lightgbm-with-select-k-best-on-tfidf?scriptVersionId=2642223 0.9785
    # 18: https://www.kaggle.com/pranav84/training-weak-learner-with-pr-fane-dict-lookup?scriptVersionId=2731592 0.9773
    # 19: https://www.kaggle.com/johnfarrell/tfidf-3layers-mlp-from-mercari 0.9774
    # 20: https://www.kaggle.com/thec03u5/nbsvm/notebook 0.9772
    # 21: https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams?scriptVersionId=2611198 0.9792

    subnums = [1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18,19,20,21]
    subs, oofs = get_subs(subnums)

    # Engineer features
    feature_functions = [len, asterix_freq, questionmark_freq, hashtag_freq, dot_freq, uppercase_freq, raw_word_num, cap_word_freq,
    nb_fk, nb_you, nb_mother, has_http, has_mail, badword_freq, ant_slash_n, nb_sk, nb_dk, nb_ng, start_with_columns, has_timestamp,
    has_date_long, has_date_short, has_emphasize_equal, has_emphasize_quotes, num_emoji, star_freq]
    features = [f.__name__ for f in feature_functions]
    F_train = engineer_features(train[INPUT_COLUMN], feature_functions)
    F_test = engineer_features(test[INPUT_COLUMN], feature_functions)

    X_train = np.hstack([F_train[features].as_matrix(), oofs])
    X_test = np.hstack([F_test[features].as_matrix(), subs])    

    import lightgbm as lgb
    stacker = lgb.LGBMClassifier(max_depth=3, metric="auc", n_estimators=125, num_leaves=10, boosting_type="gbdt", learning_rate=0.1,
        feature_fraction=0.45, colsample_bytree=0.45, bagging_fraction=0.8, bagging_freq=5, reg_lambda=0.2)

    # Fit and submit
    scores = []
    for label in LABELS:
        print(label)
        score = cross_val_score(stacker, X_train, train[label], cv=5, scoring='roc_auc')
        print("AUC:", score)
        scores.append(np.mean(score))
        stacker.fit(X_train, train[label])
        sub[label] = stacker.predict_proba(X_test)[:,1]
    print("CV score:", np.mean(scores))

    sub.to_csv("blend_otherModels.csv", index = False)
