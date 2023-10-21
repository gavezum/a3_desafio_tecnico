import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
import kds.metrics as kds

class ModelAssessment:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.pred_prob = self._predict_proba(X)
        self.metrics_table = pd.DataFrame(columns = ['Avg_precision',
                                                     'Roc_AUC'])
        
    def _predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]

    def _calculate_avg_prec(self, X, y):
        y_pred = self._predict_proba(X)
        return round(average_precision_score(y, y_pred),3)
    
    def _calculate_roc_auc(self, X, y):
        y_pred = self._predict_proba(X)
        return round(roc_auc_score(y, y_pred),3)

    def compute_shap_values(self):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X)
        shap.summary_plot(shap_values, self.X, max_display = 10)

    def plot_feat_importance(self):
        feature_importance = (pd.DataFrame.from_dict(self.model.get_booster().get_score(importance_type = 'weight'),
                                                    orient='index')
                             .sort_values(by=0))
        self.feature_importance = feature_importance
        
        feature_importance.plot(kind='barh', figsize=(10,20)).legend(loc='lower right')
        plt.show()

    def kds(self):
        kds.report(self.y, self.pred_prob)
     
    def _calculates_edges(self, array, deciles):
        percentiles = np.percentile(array,
                                   deciles)
        percentiles[0] = 0
        return percentiles
    
    def _calculate_deciles(self):
        deciles = pd.cut(self.pred_prob, 
                                       bins = self._calculates_edges(self.pred_prob,
                                                                     [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                                                                    ),
                                       labels = range(1,11)
                        )        
        return deciles
    
    def calculate_calibration(self):
        df = self.X.copy()
        df['pred_prob'] = self._predict_proba(df)
        df['target'] = self.y
        df['classe_churn'] = self._calculate_deciles()
        calibr_df = df.groupby('classe_churn')[['pred_prob','target']].mean()
        return df, calibr_df
    
    def build_metric_table(self, X_train, y_train):
        self.metrics_table.loc['train','Avg_precision'] = self._calculate_avg_prec(X_train, y_train)
        self.metrics_table.loc['train','Roc_AUC'] = self._calculate_roc_auc(X_train, y_train)
        
        self.metrics_table.loc['test','Avg_precision'] = self._calculate_avg_prec(self.X, self.y)
        self.metrics_table.loc['test','Roc_AUC'] = self._calculate_roc_auc(self.X, self.y)