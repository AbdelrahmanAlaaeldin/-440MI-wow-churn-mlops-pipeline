import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def train_model(df: pd.DataFrame, target_col: str):
    """
    Trains an {{model_name}} model and logs with MLflow.

    Args:
        df (pd.DataFrame): Feature dataset.
        target_col (str): Name of the target column.
    """
    #### TODO


    with mlflow.start_run():
            # Train model
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            rec = recall_score(y_test, preds)

            # Log params, metrics, and model
            mlflow.log_param("n_estimators", 300)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("recall", rec)
            mlflow.xgboost.log_model(model, "model")

            # 🔑 Log dataset so it shows in MLflow UI
            train_ds = mlflow.data.from_pandas(df, source="training_data")
            mlflow.log_input(train_ds, context="training")

            print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")