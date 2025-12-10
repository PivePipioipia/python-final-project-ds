"""
Module Model Trainer

Module này chứa class ModelTrainer để huấn luyện và tối ưu các mô hình học máy
cho các bài toán hồi quy (regression) trên dữ liệu dạng bảng (tabular).
Hỗ trợ RandomForest, XGBoost, và LightGBM với Optuna hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
import yaml
import logging
import joblib
import json
import os
from pathlib import Path
from datetime import datetime

# Scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# XGBoost và LightGBM
import xgboost as xgb
import lightgbm as lgb

# Optuna cho hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# Setup logging
Path("results/logs").mkdir(parents=True, exist_ok=True)

# Tạo logger riêng cho module này
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tạo formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler cho training.log
training_file_handler = logging.FileHandler('results/logs/training.log', encoding='utf-8')
training_file_handler.setLevel(logging.INFO)
training_file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Thêm handlers vào logger (chỉ nếu chưa có)
if not logger.handlers:
    logger.addHandler(training_file_handler)
    logger.addHandler(console_handler)

# Tắt logging của Optuna để giảm noise
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """
    Class để huấn luyện và tối ưu mô hình học máy cho dự đoán revenue.
    
    Class hỗ trợ:
    - Ba thuật toán: RandomForest, XGBoost, LightGBM
    - Hyperparameter optimization bằng Optuna (TPE algorithm)
    - Cross-validation để đánh giá robust
    - Logging chi tiết quá trình training
    - Lưu và load models
    - So sánh performance giữa các models
    
    Attributes:
        config (dict): Configuration từ file config.yaml
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features  
        y_train (np.ndarray): Training target
        y_test (np.ndarray): Test target
        models (Dict): Dictionary chứa các trained models
        results (Dict): Dictionary chứa kết quả đánh giá
        best_model_name (str): Tên model tốt nhất
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model_training']
        self.random_state = self.model_config['random_state']
        
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        self.y_train_log: Optional[np.ndarray] = None
        self.y_test_log: Optional[np.ndarray] = None
        
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
                
        logger.info("ModelTrainer đã được khởi tạo thành công")
    
    @property
    def best_model(self) -> Optional[Any]:
        """
        Trả về instance của model tốt nhất (nếu đã xác định),
        giúp code bên ngoài truy cập nhanh mà không cần dò dict.
        """
        if self.best_model_name is None:
            return None
        return self.models.get(self.best_model_name)


    def load_data(
        self, 
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> None:
        """
        Load dữ liệu đã được preprocess vào trainer.
        Dữ liệu đầu vào đã được xử lý hoàn chỉnh, không cần transform thêm.
        
        Args:
            X_train (np.ndarray): Training features
            X_test (np.ndarray): Test features
            y_train (np.ndarray): Training target (log-transformed)
            y_test (np.ndarray): Test target (log-transformed)
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train_log = y_train
        self.y_test_log = y_test
        
        # Inverse transform để có cả version gốc cho evaluation
        self.y_train = np.expm1(y_train)
        self.y_test = np.expm1(y_test)
        
        logger.info(
            f"Đã load data - Train: {X_train.shape}, Test: {X_test.shape}"
        )
        logger.info(
            f"Target range (original): "
            f"[${self.y_train.min():,.0f}, ${self.y_train.max():,.0f}]"
        )
        logger.info(
            f"Target range (log): "
            f"[{self.y_train_log.min():.2f}, {self.y_train_log.max():.2f}]"
        )
    
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        test_size: Optional[float] = None
    ) -> None:
        """
        Chia dữ liệu thành train và test sets.
        
        Args:
            X (np.ndarray): Features array
            y (np.ndarray): Target array
            test_size (float, optional): Tỷ lệ test set. Mặc định từ config.
        """
        if test_size is None:
            test_size = self.config['preprocessing']['test_size']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=self.random_state
        )

        if y_is_log:
            self.y_train_log = y_train
            self.y_test_log = y_test
            self.y_train = np.expm1(y_train)
            self.y_test = np.expm1(y_test)
        else:
            self.y_train = y_train
            self.y_test = y_test
            self.y_train_log = np.log1p(y_train)
            self.y_test_log = np.log1p(y_test)
        
        self.X_train = X_train
        self.X_test = X_test
        
        logger.info(
            f"Đã split data - Train: {self.X_train.shape}, Test: {self.X_test.shape}"
        )
    
    def _evaluate_model(
        self, 
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_test_log: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Đánh giá model trên test set với nhiều metrics.
        
        """
        # Model predict log(revenue)
        y_pred_log = model.predict(X_test)
        
        # Inverse transform về scale gốc
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0)
        
        # Metrics trên scale gốc
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        # Metrics trên log scale (để đánh giá model)
        if y_test_log is not None:
            rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
            mae_log = mean_absolute_error(y_test_log, y_pred_log)
            r2_log = r2_score(y_test_log, y_pred_log)
            
            metrics['RMSE_log'] = rmse_log
            metrics['MAE_log'] = mae_log
            metrics['R2_log'] = r2_log
        
        return metrics

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Phương thức public để trả về toàn bộ kết quả đánh giá hiện có.
        Thỏa yêu cầu đề: có hàm evaluate() rõ ràng.

        Returns:
            Dict[model_name, metrics_dict]
        """
        if not self.results:
            raise RuntimeError("Chưa có model nào được train để evaluate.")
        return self.results


    def _get_search_space(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Helper method để lấy search space dựa trên model name từ config.
        """
        config = self.model_config['models'][model_name]['param_space']
        params = {}

        if model_name == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', config['n_estimators'][0], config['n_estimators'][1]),
                'max_depth': trial.suggest_int('max_depth', config['max_depth'][0], config['max_depth'][1]),
                'min_samples_split': trial.suggest_int('min_samples_split', config['min_samples_split'][0], config['min_samples_split'][1]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', config['min_samples_leaf'][0], config['min_samples_leaf'][1]),
                'max_features': trial.suggest_categorical('max_features', config['max_features']),
                'random_state': self.random_state,
                'n_jobs': -1
            }
        elif model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', config['n_estimators'][0], config['n_estimators'][1]),
                'max_depth': trial.suggest_int('max_depth', config['max_depth'][0], config['max_depth'][1]),
                'learning_rate': trial.suggest_float('learning_rate', config['learning_rate'][0], config['learning_rate'][1]),
                'subsample': trial.suggest_float('subsample', config['subsample'][0], config['subsample'][1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree', config['colsample_bytree'][0], config['colsample_bytree'][1]),
                'min_child_weight': trial.suggest_int('min_child_weight', config['min_child_weight'][0], config['min_child_weight'][1]),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': 0
            }
        elif model_name == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', config['n_estimators'][0], config['n_estimators'][1]),
                'max_depth': trial.suggest_int('max_depth', config['max_depth'][0], config['max_depth'][1]),
                'learning_rate': trial.suggest_float('learning_rate', config['learning_rate'][0], config['learning_rate'][1]),
                'num_leaves': trial.suggest_int('num_leaves', config['num_leaves'][0], config['num_leaves'][1]),
                'min_child_samples': trial.suggest_int('min_child_samples', config['min_child_samples'][0], config['min_child_samples'][1]),
                'subsample': trial.suggest_float('subsample', config['subsample'][0], config['subsample'][1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree', config['colsample_bytree'][0], config['colsample_bytree'][1]),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': -1
            }
        
        return params

    def _optimize_model(self, model_class: Any, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Hàm generic để tối ưu hóa bất kỳ model nào sử dụng Optuna.
        """
        logger.info(f"Bắt đầu optimize {model_name.upper()} với Optuna...")

        def objective(trial):
            params = self._get_search_space(trial, model_name)
            model = model_class(**params)
            
            scores = cross_val_score(
                model, 
                self.X_train, 
                self.y_train_log,
                cv=self.model_config['cv_folds'],
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            return -scores.mean()

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )

        study.optimize(
            objective,
            n_trials=self.model_config['optuna']['n_trials'],
            timeout=self.model_config['optuna']['timeout'],
            show_progress_bar=True
        )

        best_params = study.best_params
        # Add lại các tham số cố định (vì study.best_params chỉ chứa các tham số search)
        best_params['random_state'] = self.random_state
        best_params['n_jobs'] = -1
        if model_name == 'xgboost':
            best_params['verbosity'] = 0
        elif model_name == 'lightgbm':
            best_params['verbosity'] = -1

        best_model = model_class(**best_params)
        best_model.fit(self.X_train, self.y_train_log)

        logger.info(f"{model_name.upper()} - Best RMSE: {study.best_value:.2f}")
        logger.info(f"{model_name.upper()} - Best params: {best_params}")

        return best_model, best_params

    def _optimize_random_forest(self) -> Tuple[Any, Dict[str, Any]]:
        return self._optimize_model(RandomForestRegressor, 'random_forest')

    def _optimize_xgboost(self) -> Tuple[Any, Dict[str, Any]]:
        return self._optimize_model(xgb.XGBRegressor, 'xgboost')

    def _optimize_lightgbm(self) -> Tuple[Any, Dict[str, Any]]:
        return self._optimize_model(lgb.LGBMRegressor, 'lightgbm')
    
    def optimize_params(self, model_name: str) -> Dict[str, Any]:
        """
        Tối ưu siêu tham số cho một model cụ thể (theo yêu cầu đề: optimize_params()).

        Args:
            model_name (str): 'random_forest' | 'xgboost' | 'lightgbm'

        Returns:
            Dict[str, Any]: best hyperparameters cho model đó
        """
        if self.X_train is None:
            raise RuntimeError("Chưa có dữ liệu. Gọi load_data() hoặc split_data() trước.")

        valid_models = ['random_forest', 'xgboost', 'lightgbm']
        if model_name not in valid_models:
            raise ValueError(f"Model name phải là một trong: {valid_models}")

        if model_name == 'random_forest':
            model, params = self._optimize_random_forest()
        elif model_name == 'xgboost':
            model, params = self._optimize_xgboost()
        else:
            model, params = self._optimize_lightgbm()

        self.models[model_name] = model
        self.best_params[model_name] = params

        return params


    def train_model(self, model_name: str) -> None:

        if self.X_train is None:
            raise RuntimeError("Chưa load data. Vui lòng gọi load_data() hoặc split_data() trước.")
        
        valid_models = ['random_forest', 'xgboost', 'lightgbm']
        if model_name not in valid_models:
            raise ValueError(f"Model name phải là một trong: {valid_models}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        start_time = datetime.now()
        
        if model_name == 'random_forest':
            model, params = self._optimize_random_forest()
        elif model_name == 'xgboost':
            model, params = self._optimize_xgboost()
        elif model_name == 'lightgbm':
            model, params = self._optimize_lightgbm()
        
        metrics = self._evaluate_model(
            model, 
            self.X_test, 
            self.y_test,
            self.y_test_log
        )
        
        self.models[model_name] = model
        self.best_params[model_name] = params
        self.results[model_name] = metrics
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"\n{model_name.upper()} - Test Set Results:")
        logger.info(f"  RMSE: ${metrics['RMSE']:,.2f}")
        logger.info(f"  MAE:  ${metrics['MAE']:,.2f}")
        logger.info(f"  R2:   {metrics['R2']:.4f}")
        logger.info(f"  MAPE: {metrics['MAPE']:.2f}%")
        logger.info(f"Training time: {training_time:.2f} seconds")
    
    def train_all_models(self) -> None:
        """
        Train tất cả các models (RandomForest, XGBoost, LightGBM).
        """
        logger.info("\n" + "="*60)
        logger.info("BẮT ĐẦU TRAINING TẤT CẢ MODELS")
        logger.info("="*60 + "\n")
        
        models_to_train = ['random_forest', 'xgboost', 'lightgbm']
        
        for model_name in models_to_train:
            try:
                self.train_model(model_name)
            except Exception as e:
                logger.error(f"Lỗi khi training {model_name}: {e}")
        
        self._determine_best_model()
        
        logger.info("\n" + "="*60)
        logger.info("ĐÃ HOÀN THÀNH TRAINING TẤT CẢ MODELS")
        logger.info("="*60 + "\n")
    
    def _determine_best_model(self) -> None:
        """
        Xác định model tốt nhất dựa trên RMSE.
        """
        if not self.results:
            logger.warning("Chưa có kết quả nào để so sánh")
            return
        
        best_model = min(self.results.items(), key=lambda x: x[1]['RMSE'])
        self.best_model_name = best_model[0]
        
        logger.info(f"\nBEST MODEL: {self.best_model_name.upper()}")
        logger.info(f"  RMSE: ${best_model[1]['RMSE']:,.2f}")
        logger.info(f"  R2:   {best_model[1]['R2']:.4f}")
    
    def compare_models(self) -> pd.DataFrame:
        """
        So sánh performance của tất cả các models.
        
        Returns:
            pd.DataFrame: DataFrame chứa kết quả so sánh
        """
        if not self.results:
            raise RuntimeError("Chưa có model nào được train")
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.sort_values('RMSE')
        
        logger.info("\n" + "="*60)
        logger.info("SO SÁNH MODELS")
        logger.info("="*60)
        print(comparison_df.to_string())
        
        return comparison_df
    
    def save_model(self, model_name: str, filepath: str) -> None:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} chưa được train")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Đã lưu {model_name} vào: {filepath}")
    
    def save_best_model(self, filepath: str) -> None:
        if self.best_model_name is None:
            raise RuntimeError("Chưa xác định được best model")
        
        self.save_model(self.best_model_name, filepath)
    
    def save_all_models(self, directory: str = "models") -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for model_name in self.models:
            filepath = f"{directory}/{model_name}.pkl"
            self.save_model(model_name, filepath)
    
    def save_results(self, filepath: str = "results/experiments.csv") -> None:
        if not self.results:
            logger.warning("Chưa có kết quả nào để lưu")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.results).T
        df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath, index_col=0)
            df = pd.concat([existing_df, df])
        
        df.to_csv(filepath)
        logger.info(f"Đã lưu kết quả vào: {filepath}")
    
    def save_best_params(self, filepath: str = "results/best_params.json") -> None:
        if not self.best_params:
            logger.warning("Chưa có best params nào để lưu")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        logger.info(f"Đã lưu best params vào: {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> Any:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
        
        model = joblib.load(filepath)
        logger.info(f"Đã load model từ: {filepath}")
        return model
    
    def get_feature_importance(
        self, 
        model_name: str,
        feature_names: List[str],
        top_n: int = 15
    ) -> pd.DataFrame:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} chưa được train")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise AttributeError(f"Model {model_name} không có feature_importances_")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def __repr__(self) -> str:
        """String representation của ModelTrainer."""
        n_models = len(self.models)
        best = self.best_model_name or "None"
        return (
            f"ModelTrainer("
            f"n_models_trained={n_models}, "
            f"best_model={best}"
            f")"
        )

    def evaluate_external_model(
        self, 
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_test_log: Optional[np.ndarray] = None,
        preprocessor: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        PUBLIC method để evaluate model từ bên ngoài
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target (scale GỐC)
            y_test_log: Test target (log scale), optional
            preprocessor: DataPreprocessor instance, optional
            
        Returns:
            Dictionary of metrics
        """
        # Nếu chỉ có y_test (scale gốc), tạo y_test_log
        if y_test_log is None and y_test.max() > 50:
            y_test_log = np.log1p(y_test)
        
        # Nếu chỉ có y_test_log, inverse transform
        if y_test_log is not None and y_test.max() < 50:
            if preprocessor is not None:
                y_test = preprocessor.inverse_transform_target(y_test_log)
            else:
                y_test = np.expm1(y_test_log)
        
        return self._evaluate_model(model, X_test, y_test, y_test_log)



if __name__ == "__main__":
    X_train = pd.read_csv("data/processed/X_train.csv").values
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    X_test = pd.read_csv("data/processed/X_test.csv").values
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    
    trainer = ModelTrainer()
    trainer.load_data(X_train, X_test, y_train, y_test)
    
    trainer.train_all_models()
    
    comparison = trainer.compare_models()
    
    trainer.save_all_models()
    trainer.save_results()
    trainer.save_best_params()