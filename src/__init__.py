"""
Movie Revenue Prediction Package

Package này chứa toàn bộ các module cho dự án dự đoán doanh thu phim, bao gồm data loading, preprocessing, model training, và visualization.

Modules:
    - data_loader: Thu thập dữ liệu từ TMDb API
    - preprocessing: Tiền xử lý và feature engineering
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