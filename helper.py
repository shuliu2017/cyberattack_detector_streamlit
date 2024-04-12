
## pipeline for preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    '''
    input pd.dataframe, output frequency of each category in the series.
    '''
    def __init__(self):
        pass
    
    def fit(self, X):
        self.columns = X.columns
        self.lst = []
        for col in X.columns:
            self.lst.append(Counter(X[col].values))
        
    def transform(self, X, y=None):
        Xcopy = X.copy()
        for i, col in enumerate(X.columns):
            Xcopy[col] = Xcopy[col].map(self.lst[i])
        return Xcopy
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def get_feature_names_out(self):
        return self.columns
    
    
## build a class to transform np array to df
class ArrayToDf(BaseEstimator, TransformerMixin):
    '''
    input np.array, output pd.DataFrame
    '''
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X):
        pass
    
    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.columns)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    def get_feature_names_out(self):
        return self.columns




class cutCatTranformer(BaseEstimator, TransformerMixin):
    '''
    input pd.dataframe, cut categories with count less than threshold into one category 'Others'.
    '''
    
    def __init__(self, threshold=3):
        self.threshold = threshold
        
    def fit(self, X):
        self.columns = X.columns
        self.lst = []
        for col in X.columns:
            self.lst.append(Counter(X[col].values))
        
        
    def transform(self, X, y=None):
        Xcopy = X.copy()
        for i, col in enumerate(X.columns):
            Xcopy[col] = X[col].mask(X[col].map(self.lst[i]) < self.threshold, 'Others')
        return Xcopy
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
def cut_cat(col, threshold=3):
    """
    This function cut categories with count less than threshold into one category 'others'.
    """
    freq = col.value_counts()
    col_new = col.mask(col.map(freq) < threshold, 'Other')
    return col_new


## get the columnnames after preprocessing

def get_feature_names(column_transformer):
    
    feature_names = []
    
    for name, pipe, features in column_transformer.transformers_:
        if name == 'num':
            feature_names.extend(num_features)
        elif name == 'dummy':
            ohe = pipe.named_steps['onehot']
            feature_names.extend(ohe.get_feature_names_out())
        elif name == 'freq':
            feature_names.extend(pipe.named_steps['ArrayToDf'].get_feature_names_out())
        else:
            pass
    return feature_names



# a function to plot the confusion matrix
def plot_confusion_matrix(cm, labels):
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title="Confusion matrix",
           ylabel="True label",
           xlabel="Predicted label")
    # rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # loop over the data and create text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    return fig

from sklearn.metrics import precision_recall_curve

def plot_precision_recall_curve(y_test, estimator, X_test):
    
    y_test_prob = estimator.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_prob)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    
    
    
def plot_classification_report(cr, labels):
    
    labels = [str(label) for label in labels.tolist()]
    fig, axes = plt.subplots(1,3, figsize=(15, 5))
    axes = axes.flatten()
    prec,recall,f1 = [],[],[]
    for label in labels+['macro avg', 'weighted avg']:
        prec.append(cr[label]['precision'] )
        recall.append(cr[label]['recall'])
        f1.append(cr[label]['f1-score'])
    axes[0].barh(labels+['macro avg', 'weighted avg'], prec)
    axes[0].set_title('Precision')
    axes[1].barh(labels+['macro avg', 'weighted avg'], recall)
    axes[1].set_title('Recall')
    axes[2].barh(labels+['macro avg', 'weighted avg'], f1)
    axes[2].set_title('F1-score')
    plt.tight_layout()
    plt.show()

    
import sklearn.metrics as metrics

# a function to compute the confusion matrix and the classification report
def compute_metrics(clf, X_test, y_test, labels):
    # predict the test data using the classifier
    y_pred = clf.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred, labels=labels)
    cr = metrics.classification_report(y_test, y_pred, labels=labels, output_dict=True)
    return cm, cr

from numpy import dtype

data_type = {'arp.opcode': dtype('float64'),
 'arp.hw.size': dtype('float64'),
 'icmp.checksum': dtype('float64'),
 'icmp.seq_le': dtype('float64'),
 'icmp.unused': dtype('float64'),
 'http.content_length': dtype('float64'),
 'http.request.method': dtype('O'),
 'http.referer': dtype('O'),
 'http.request.version': dtype('O'),
 'http.response': dtype('float64'),
 'http.tls_port': dtype('float64'),
 'tcp.ack': dtype('float64'),
 'tcp.ack_raw': dtype('float64'),
 'tcp.checksum': dtype('float64'),
 'tcp.connection.fin': dtype('float64'),
 'tcp.connection.rst': dtype('float64'),
 'tcp.connection.syn': dtype('float64'),
 'tcp.connection.synack': dtype('float64'),
 'tcp.flags': dtype('float64'),
 'tcp.flags.ack': dtype('float64'),
 'tcp.len': dtype('float64'),
 'tcp.seq': dtype('float64'),
 'udp.stream': dtype('float64'),
 'udp.time_delta': dtype('float64'),
 'dns.qry.name': dtype('float64'),
 'dns.qry.name.len': dtype('O'),
 'dns.qry.qu': dtype('float64'),
 'dns.qry.type': dtype('float64'),
 'dns.retransmission': dtype('float64'),
 'dns.retransmit_request': dtype('float64'),
 'dns.retransmit_request_in': dtype('float64'),
 'mqtt.conack.flags': dtype('O'),
 'mqtt.conflag.cleansess': dtype('float64'),
 'mqtt.conflags': dtype('float64'),
 'mqtt.hdrflags': dtype('float64'),
 'mqtt.len': dtype('float64'),
 'mqtt.msg_decoded_as': dtype('float64'),
 'mqtt.msgtype': dtype('float64'),
 'mqtt.proto_len': dtype('float64'),
 'mqtt.protoname': dtype('O'),
 'mqtt.topic': dtype('O'),
 'mqtt.topic_len': dtype('float64'),
 'mqtt.ver': dtype('float64'),
 'mbtcp.len': dtype('float64'),
 'mbtcp.trans_id': dtype('float64'),
 'mbtcp.unit_id': dtype('float64'),
 'Attack_label': dtype('int64'),
 'Attack_type': dtype('O')}
