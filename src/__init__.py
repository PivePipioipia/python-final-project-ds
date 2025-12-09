"""
Movie Revenue Prediction Package

Modules:
    - base_loader: Định nghĩa giao diện chung cho mọi DataLoader trong pipeline
    - data_loader: Thu thập dữ liệu từ TMDb API
    - base_preprocessor: Định nghĩa giao diện chung cho mọi Preprocessor trong pipeline
    - preprocessing: Tiền xử lý và feature engineering (basic)
    - preprocessing_v2: Tiền xử lý và feature engineering (advanced)
    - model_trainer: Huấn luyện và tối ưu mô hình học máy
    - visualizer: Trực quan hóa dữ liệu và kết quả

Author: 
Date: 
"""

__version__ = "1.0.0"
__author__ = "Data Science Student"

from .data_loader import TMDbDataLoader
from .preprocessing import DataPreprocessor
from .model_trainer import ModelTrainer
from .visualizer import Visualizer

__all__ = [
    "TMDbDataLoader",
    "DataPreprocessor", 
    "ModelTrainer",
    "Visualizer"
]