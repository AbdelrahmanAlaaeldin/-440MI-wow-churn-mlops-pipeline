import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def tune_model(X, y):
    """
    Tunes an {{model_name}} model using Optuna.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """

    def objective(trial):
        params = {
            #### TODO
        }
        model = XGBClassifier(**params) #### TODO Change model_name
        scores = cross_val_score(model, X, y, cv=3, scoring="recall")
        return scores.mean()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best Params:", study.best_params)
    return study.best_params