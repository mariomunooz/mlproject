import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path : str = os.path.join('artifacts', 'preprocessor.pkl')
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        
        '''This function is responsible for data transformation'''
        try:
            
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
                ]
            
            # This pipeline is used to preprocess numerical data.
            # It consists of two steps:
            # 1. 'imputer': This step handles missing values by replacing them with the median value 
            # of the column.
            # 2. 'scaler': After imputation, this step standardizes the numerical features by scaling them 
            # to have mean 0 and variance 1 (z-score normalization).
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                    ]
                )
            
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder())
                    ]
                )
            
            logging.info("Created the numerical and categorical pipelines")
            
            
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                    ]
                )
            
            logging.info("Created the preprocessor object")
            
            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read the train and test data completed")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            
            target_column_name = 'math_score'
            
            numerical_columns = ['reading_score', 'writing_score']
            
            input_feature_train_df = train_df.drop(columns = [target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns = [target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            
            # Applies preprocessing transformations to both the training and test input feature data.
            # It then combines the transformed input features with the target labels, resulting in arrays where 
            # each element contains the scaled features followed by the target label as the last element.

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Applied preprocessing object on the training dataframe and testing dataframe completed")
            
            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj

            )
            
            logging.info(f"Saved preprocessing object.")
            
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
            
            
            
            
            
        except Exception as e:
            raise CustomException(e, sys)