import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

        if model_type == "logistic":
            self.model = LogisticRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Select features from DataFrame and apply standard scaling.

        Args:
            df (pd.DataFrame): Input data including features and target.

        Returns:
            np.ndarray: Scaled feature matrix ready for modeling.
        """
        X = df.drop(columns=['target', 'fwd_return'])
        # Fit scaler on data and transform
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def train(self, df: pd.DataFrame):
        """
        Train the logistic regression model using features and target label.

        Args:
            df (pd.DataFrame): DataFrame including features and target column 'target'.
        """
        X = self.preprocess_data(df)
        y = df['target']
        self.model.fit(X, y)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probability estimates for the positive class.

        Args:
            df (pd.DataFrame): DataFrame with features to predict.

        Returns:
            np.ndarray: Array of probabilities for class 1.
        """
        X = self.preprocess_data(df)
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
        X_scaled = self.preprocess_data(df)
        return self.model.predict(X_scaled)
