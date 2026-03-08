import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

class MLModel:
    """
    A simple wrapper class for machine learning models applied to trading features.
    
    Currently supports logistic regression with standard scaling preprocessing.
    
    Attributes:
        model_type (str): Type of model to use. Only 'logistic' supported now.
        scaler (StandardScaler): Scaler used to normalize feature data.
        model (sklearn model): The ML model instance.
    """

    def __init__(self, model_type="logistic"):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.feature_cols_ = None

        if model_type == "logistic":
            self.model = LogisticRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _select_feature_columns(self, df: pd.DataFrame) -> list[str]:
        excluded_cols = {'Close', 'High', 'Low', 'Open', 'Volume', 'fwd_return', 'target'}
        return [col for col in df.columns if col not in excluded_cols]

    def preprocess_data(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Select features from DataFrame and apply standard scaling.

        Args:
            df (pd.DataFrame): Input data including features and target.
            fit (bool): If True, fit scaler on this data; otherwise transform with fitted scaler.

        Returns:
            np.ndarray: Scaled feature matrix ready for modeling.
        """
        if fit:
            self.feature_cols_ = self._select_feature_columns(df)
            X = df[self.feature_cols_]
            return self.scaler.fit_transform(X)

        if self.feature_cols_ is None:
            raise NotFittedError("Model is not fitted yet. Call train() before predict().")

        missing_cols = [col for col in self.feature_cols_ if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        X = df[self.feature_cols_]
        return self.scaler.transform(X)

    def train(self, df: pd.DataFrame):
        """
        Train the logistic regression model using features and target label.

        Args:
            df (pd.DataFrame): DataFrame including features and target column 'target'.
        """
        X = self.preprocess_data(df, fit=True)
        y = df['target']
        if y.nunique() < 2:
            raise ValueError("Target must contain at least two classes for logistic regression.")
        self.model.fit(X, y)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probability estimates for the positive class.

        Args:
            df (pd.DataFrame): DataFrame with features to predict.

        Returns:
            np.ndarray: Array of probabilities for class 1.
        """
        X = self.preprocess_data(df, fit=False)
        probs = self.model.predict_proba(X)[:, 1]
        return probs

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels (0 or 1) based on model threshold.

        Args:
            df (pd.DataFrame): DataFrame with features to predict.

        Returns:
            np.ndarray: Predicted class labels.
        """
        X_scaled = self.preprocess_data(df, fit=False)
        return self.model.predict(X_scaled)
