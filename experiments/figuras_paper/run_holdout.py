import sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0,'.')
from original_model import RegionalMAPEM
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
S=42
raw=pd.read_csv('telco_raw.csv'); raw.columns=raw.columns.str.strip()
tr=pd.read_csv('train_index.csv')['train_index'].astype(int).to_numpy()
te=pd.read_csv('test_index.csv')['test_index'].astype(int).to_numpy()
X=raw.drop(columns=['customerID','Churn']); y=raw['Churn'].map({'No':0,'Yes':1}).astype(int).to_numpy()
NUMERIC=["tenure","MonthlyCharges","TotalCharges","SeniorCitizen","AvgChargePerMonth"]
SERVICE=["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
CAT=[c for c in list(X.columns)+["AvgChargePerMonth"] if c not in NUMERIC]
class HF(TransformerMixin,BaseEstimator):
    def fit(self,X,y=None): return self
    def transform(self,X):
        o=X.copy(); t=pd.to_numeric(o["TotalCharges"],errors="coerce"); o["TotalCharges"]=t.fillna(0.0)
        for c in SERVICE: o[c]=o[c].replace("No internet service","No")
        o["MultipleLines"]=o["MultipleLines"].replace("No phone service","No")
        a=o["MonthlyCharges"].to_numpy(float).copy()
        np.divide(o["TotalCharges"].to_numpy(float),o["tenure"].to_numpy(float),out=a,where=o["tenure"].to_numpy()>0)
        o["AvgChargePerMonth"]=a; return o
CAT=[c for c in X.columns if c not in NUMERIC]
pre=Pipeline([("f",HF()),("p",ColumnTransformer([("n",StandardScaler(),NUMERIC),("c",OneHotEncoder(handle_unknown="ignore",drop="first",sparse_output=False),CAT)],remainder="drop",sparse_threshold=0.0))])
Xtr=pre.fit_transform(X.iloc[tr]); Xte=pre.transform(X.iloc[te]); ytr=y[tr]; yte=y[te]
print(Xtr.shape,Xte.shape,ytr.mean(),yte.mean())
CONT=[0,1,2,4]
EXP=[("LR",LogisticRegression(max_iter=3000)),
("LR_splines",Pipeline([("s",ColumnTransformer([("s",SplineTransformer(n_knots=5),CONT)],remainder="passthrough")),("l",LogisticRegression(C=0.5,max_iter=4000))])),
("RF",RandomForestClassifier(n_estimators=300,min_samples_leaf=5,n_jobs=-1,random_state=S)),
("XGB",XGBClassifier(n_estimators=300,learning_rate=0.03,max_depth=3,subsample=0.8,colsample_bytree=0.8,random_state=S,verbosity=0)),
("HistGB",HistGradientBoostingClassifier(learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=40,random_state=S)),
("kNN",KNeighborsClassifier(n_neighbors=50,weights="distance")),
("CatBoost",CatBoostClassifier(iterations=400,learning_rate=0.05,depth=4,verbose=0,random_seed=S,thread_count=4)),
("LightGBM",LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,random_state=S,verbose=-1)),
("NaiveBayes",GaussianNB()),
("Tree",DecisionTreeClassifier(max_depth=6,min_samples_leaf=20,random_state=S)),
("SVM",SVC(C=1.0,gamma="scale",random_state=S))]
t=time.time()
m=RegionalMAPEM(experts=EXP,n_regions=4,prior_strength=20,inner_cv=3,max_iter=5000,random_state=S)
m.fit(Xtr,ytr)
print("fit s",time.time()-t,"conv",m.converged_)
P=m.expert_proba(Xte); p=m.predict_proba(Xte)[:,1]
out=pd.DataFrame(P,columns=m.expert_names_); out["RegionalMAPEM"]=p; out["y"]=yte; out.to_csv('holdout_probs_reconstruida.csv',index=False)
for n in out.columns[:-1]: print(n, round(log_loss(yte,out[n]),4), round(roc_auc_score(yte,out[n]),4))
