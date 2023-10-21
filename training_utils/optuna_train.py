import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score


class OptunaXGBWithCV:
    def __init__(self, n_trials=100, n_splits=5, top_k_models=3, random_state=123):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.top_k_models = top_k_models
        self.random_state = random_state
        self.experiments = pd.DataFrame(columns = ['params',
                                                   'train_score',
                                                   'test_score',
                                                   'overfit'])
    
    def _Objective(self, trial):
        # Define the hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'reg_alpha': trial.suggest_float('subsample', 0, 1.0),
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        # Create and train the model with current hyperparameters
        model = XGBClassifier(**params)
        
        # Perform cross-validation
        cv_pred = cross_validate(model, 
                                self.X, 
                                self.y, 
                                cv= self.n_splits, 
                                return_train_score = True, 
                                scoring='average_precision',)
        
        # Calculate the statistics in both folds
        mean_train_score = np.mean(cv_pred['train_score'])
        mean_test_score = np.mean(cv_pred['test_score'])
        overfit = mean_train_score - mean_test_score
        # Append the model and its parameters to the dataframe
        new_row_df = pd.DataFrame(
            {
                'params': [params],
                'train_score': [mean_train_score],
                'test_score': [mean_test_score],
                'overfit': [overfit]
            } 
        )
        
        # Append the run into the experiment df
        self.experiments = pd.concat([self.experiments, new_row_df], 
                                     ignore_index = True)
        
        return mean_test_score
    
    def Fit(self, X, y):
        self.X = X
        self.y = y
        
        # Perform hyperparameter search using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(self._Objective, n_trials=self.n_trials, show_progress_bar=True)
        
    def Select_best_models(self, overfit_limit):
        # Method to select the best models based on specified criteria and a limit for overfitting.
        experiment_df = self.experiments # Get the experiments data.
        experiment_df = experiment_df.reset_index().rename(columns = {'index': 'run_id'})
        
        # Check if there are any models with overfitting below the limit.
        assert len(experiment_df[experiment_df.overfit < overfit_limit]) > 0 ,f'Sem run com um overfit menor que {overfit_limit}'
        
        # Filter and sort the experiments to select the top-k models.
        experiment_df = (experiment_df[experiment_df.overfit < overfit_limit].
                         sort_values(by = 'test_score', 
                                     ascending = False).
                        iloc[0:self.top_k_models,:])
        
        return experiment_df
    
    def train_model(self, params: dict):
        # Train a model given a set of hyperparameters
        model = XGBClassifier(**params).fit(self.X, self.y)
        return model
        
        
