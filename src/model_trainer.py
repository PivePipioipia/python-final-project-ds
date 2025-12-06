"""
Module Model Trainer

Module này chứa class ModelTrainer để huấn luyện và tối ưu các mô hình học máy cho bài toán dự đoán revenue. Hỗ trợ RandomForest, XGBoost, và LightGBM 
với Optuna hyperparameter optimization.
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        """
        Khởi tạo ModelTrainer với configuration.
        
        Args:
            config_path (str): Đường dẫn đến file config.yaml
            
        Raises:
            FileNotFoundError: Nếu không tìm thấy file config
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model_training']
        self.random_state = self.model_config['random_state']
        
        # Data placeholders
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        
        # Models và results
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        
        # Tạo thư mục logs nếu chưa có
        Path("results/logs").mkdir(parents=True, exist_ok=True)
        
        logger.info("ModelTrainer đã được khởi tạo thành công")
    
    def load_data(
        self, 
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> None:
        """
        Load dữ liệu đã được preprocess vào trainer.
        
        Args:
            X_train (np.ndarray): Training features
            X_test (np.ndarray): Test features
            y_train (np.ndarray): Training target
            y_test (np.ndarray): Test target
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(
            f"Đã load data - Train: {X_train.shape}, Test: {X_test.shape}"
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
        
        logger.info(
            f"Đã split data - Train: {self.X_train.shape}, Test: {self.X_test.shape}"
        )
    
    def _evaluate_model(
        self, 
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Đánh giá model trên test set với nhiều metrics.
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            
        Returns:
            Dict[str, float]: Dictionary chứa các metrics
        """
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    def _optimize_random_forest(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Tối ưu RandomForest bằng Optuna.
        
        Returns:
            Tuple: (best_model, best_params)
        """
        logger.info("Bắt đầu optimize RandomForest với Optuna...")
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 10, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            # Train model với cross-validation
            model = RandomForestRegressor(**params)
            
            # Dùng negative RMSE làm score (vì cross_val_score maximize)
            scores = cross_val_score(
                model, 
                self.X_train, 
                self.y_train,
                cv=self.model_config['cv_folds'],
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            
            return -scores.mean()  # Return positive RMSE
        
        # Tạo study
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
        
        # Train model với best params
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        best_params['n_jobs'] = -1
        
        best_model = RandomForestRegressor(**best_params)
        best_model.fit(self.X_train, self.y_train)
        
        logger.info(f"RandomForest - Best RMSE: {study.best_value:.2f}")
        logger.info(f"RandomForest - Best params: {best_params}")
        
        return best_model, best_params
    
    def _optimize_xgboost(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Tối ưu XGBoost bằng Optuna.
        
        Returns:
            Tuple: (best_model, best_params)
        """
        logger.info("Bắt đầu optimize XGBoost với Optuna...")
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            # Train model với cross-validation
            model = xgb.XGBRegressor(**params)
            
            scores = cross_val_score(
                model,
                self.X_train,
                self.y_train,
                cv=self.model_config['cv_folds'],
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            
            return -scores.mean()
        
        # Tạo study
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
        
        # Train model với best params
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        best_params['n_jobs'] = -1
        best_params['verbosity'] = 0
        
        best_model = xgb.XGBRegressor(**best_params)
        best_model.fit(self.X_train, self.y_train)
        
        logger.info(f"XGBoost - Best RMSE: {study.best_value:.2f}")
        logger.info(f"XGBoost - Best params: {best_params}")
        
        return best_model, best_params
    
    def _optimize_lightgbm(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Tối ưu LightGBM bằng Optuna.
        
        Returns:
            Tuple: (best_model, best_params)
        """
        logger.info("Bắt đầu optimize LightGBM với Optuna...")
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': -1
            }
            
            # Train model với cross-validation
            model = lgb.LGBMRegressor(**params)
            
            scores = cross_val_score(
                model,
                self.X_train,
                self.y_train,
                cv=self.model_config['cv_folds'],
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            
            return -scores.mean()
        
        # Tạo study
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
        
        # Train model với best params
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        best_params['n_jobs'] = -1
        best_params['verbosity'] = -1
        
        best_model = lgb.LGBMRegressor(**best_params)
        best_model.fit(self.X_train, self.y_train)
        
        logger.info(f"LightGBM - Best RMSE: {study.best_value:.2f}")
        logger.info(f"LightGBM - Best params: {best_params}")
        
        return best_model, best_params
    
    def train_model(self, model_name: str) -> None:
        """
        Train một model cụ thể với Optuna optimization.
        
        Args:
            model_name (str): Tên model ('random_forest', 'xgboost', 'lightgbm')
            
        Raises:
            ValueError: Nếu model_name không hợp lệ
            RuntimeError: Nếu chưa load data
        """
        if self.X_train is None:
            raise RuntimeError("Chưa load data. Vui lòng gọi load_data() hoặc split_data() trước.")
        
        valid_models = ['random_forest', 'xgboost', 'lightgbm']
        if model_name not in valid_models:
            raise ValueError(f"Model name phải là một trong: {valid_models}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Optimize và train
        if model_name == 'random_forest':
            model, params = self._optimize_random_forest()
        elif model_name == 'xgboost':
            model, params = self._optimize_xgboost()
        elif model_name == 'lightgbm':
            model, params = self._optimize_lightgbm()
        
        # Evaluate trên test set
        metrics = self._evaluate_model(model, self.X_test, self.y_test)
        
        # Lưu kết quả
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
        
        # Xác định best model
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
        
        # Tìm model có RMSE thấp nhất
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
        """
        Lưu một model cụ thể vào file.
        
        Args:
            model_name (str): Tên model cần lưu
            filepath (str): Đường dẫn file để lưu
            
        Raises:
            ValueError: Nếu model chưa được train
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} chưa được train")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Đã lưu {model_name} vào: {filepath}")
    
    def save_best_model(self, filepath: str) -> None:
        """
        Lưu model tốt nhất vào file.
        
        Args:
            filepath (str): Đường dẫn file để lưu
        """
        if self.best_model_name is None:
            raise RuntimeError("Chưa xác định được best model")
        
        self.save_model(self.best_model_name, filepath)
    
    def save_all_models(self, directory: str = "models") -> None:
        """
        Lưu tất cả các trained models vào thư mục.
        
        Args:
            directory (str): Thư mục để lưu models
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for model_name in self.models:
            filepath = f"{directory}/{model_name}.pkl"
            self.save_model(model_name, filepath)
    
    def save_results(self, filepath: str = "results/experiments.csv") -> None:
        """
        Lưu kết quả thí nghiệm vào file CSV.
        
        Args:
            filepath (str): Đường dẫn file CSV
        """
        if not self.results:
            logger.warning("Chưa có kết quả nào để lưu")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Tạo DataFrame từ results
        df = pd.DataFrame(self.results).T
        df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Nếu file đã tồn tại, append vào
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath, index_col=0)
            df = pd.concat([existing_df, df])
        
        df.to_csv(filepath)
        logger.info(f"Đã lưu kết quả vào: {filepath}")
    
    def save_best_params(self, filepath: str = "results/best_params.json") -> None:
        """
        Lưu best parameters của tất cả models vào JSON.
        
        Args:
            filepath (str): Đường dẫn file JSON
        """
        if not self.best_params:
            logger.warning("Chưa có best params nào để lưu")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        logger.info(f"Đã lưu best params vào: {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> Any:
        """
        Load model từ file.
        
        Args:
            filepath (str): Đường dẫn file model
            
        Returns:
            Model đã được load
        """
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
        """
        Lấy feature importance của một model.
        
        Args:
            model_name (str): Tên model
            feature_names (List[str]): Danh sách tên features
            top_n (int): Số lượng top features cần lấy
            
        Returns:
            pd.DataFrame: DataFrame chứa feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} chưa được train")
        
        model = self.models[model_name]
        
        # Lấy feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise AttributeError(f"Model {model_name} không có feature_importances_")
        
        # Tạo DataFrame
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


if __name__ == "__main__":
    # Load preprocessed data
    X_train = pd.read_csv("data/processed/X_train.csv").values
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    X_test = pd.read_csv("data/processed/X_test.csv").values
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    
    # Khởi tạo trainer
    trainer = ModelTrainer()
    trainer.load_data(X_train, X_test, y_train, y_test)
    
    # Train tất cả models
    trainer.train_all_models()
    
    # So sánh models
    comparison = trainer.compare_models()
    
    # Lưu models và kết quả
    trainer.save_all_models()
    trainer.save_results()
    trainer.save_best_params()