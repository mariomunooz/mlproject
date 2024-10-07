import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models, load_params_from_yaml



@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Entered the model training method or component")
            
            X_train, y_train, X_test, y_test = (
            train_array[:,:-1], 
            train_array[:,-1], 
            test_array[:,:-1], 
            test_array[:,-1]
            )
            
            
            models = {
                "Random_Forest": RandomForestRegressor(),
                "Decision_Tree": DecisionTreeRegressor(),
                "Gradient_Boosting": GradientBoostingRegressor(),
                "Linear_Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting_Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost_Regressor": AdaBoostRegressor(),
            }
            
            
            # do hyperparameter tuning  
            params = load_params_from_yaml(os.path.join("config", "model_params.yaml"))
            model_report:dict = evaluate_models(X_train=X_train, y_train = y_train, X_test=X_test, y_test=y_test, 
                                                models=models, params=params)
            
            
            # Get the best model name and score
            best_model_name, best_model_score = max(model_report.items(), key=lambda x: x[1])

            # Get the best model from the models dictionary
            best_model = models[best_model_name]
            
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_test_predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, y_test_predicted)
            return r2_square
            
            
            
            
            
            
            
            
        except Exception as e:
            raise CustomException(e,sys)