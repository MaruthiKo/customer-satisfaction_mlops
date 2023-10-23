import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for Model
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the Model
        Args:
            X_train: Training data
            y_train: Training labels
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Train Linear Regression Model
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """
        try:
            model = LinearRegression(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Linear Regression Model Trained")
            return model
        except Exception as e:
            logging.error(f"Error in training Linear Regression Model: {e}")
            raise e
    