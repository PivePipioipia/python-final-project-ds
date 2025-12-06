"""
Module Base Preprocessor

Định nghĩa lớp trừu tượng BasePreprocessor để chuẩn hóa giao diện
cho tất cả các preprocessor trong pipeline tiền xử lý dữ liệu.

Mục tiêu:
- Đảm bảo các preprocessor có cùng "hợp đồng" phương thức (fit / transform / fit_transform)
- Giúp pipeline dễ dàng thay thế / mở rộng sang bài toán hoặc dataset khác
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List
import pandas as pd
import numpy as np


class BasePreprocessor(ABC):
    """
    Lớp trừu tượng cho mọi preprocessor trong pipeline.

    Các phương thức quan trọng:
    - fit(df): học các tham số tiền xử lý từ dữ liệu train
    - transform(df, ...): áp dụng pipeline đã được fit lên dữ liệu mới
    - fit_transform(df, ...): tiện ích = fit() + transform() cho dữ liệu huấn luyện
    - get_feature_names(): trả về danh sách tên features sau khi tiền xử lý

    Thuộc tính chung:
        name (str): Tên preprocessor (hữu ích khi log hoặc debug)
        target_col (str): Tên cột target (vd: 'revenue')
        feature_names (List[str]): Danh sách tên features sau preprocessing
        is_fitted (bool): Trạng thái preprocessor đã được fit hay chưa
    """

    def __init__(self, name: str = "base_preprocessor", target_col: str = "revenue") -> None:
        self.name = name
        self.target_col = target_col
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BasePreprocessor":
        """
        Học các tham số tiền xử lý từ dữ liệu huấn luyện
        (vd: scaler parameters, vocabulary, danh sách genres, ...)

        Args:
            df (pd.DataFrame): DataFrame huấn luyện (thường chứa cả cột target)

        Returns:
            BasePreprocessor: self (để có thể chain method nếu muốn)
        """
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        *args: Any,
        **kwargs: Any
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Áp dụng pipeline tiền xử lý đã được fit lên dữ liệu mới.

        Args:
            df (pd.DataFrame): DataFrame đầu vào (train/test/inference)

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                - X: numpy array features sau khi transform
                - y: numpy array target (nếu df có cột target), otherwise None
        """
        raise NotImplementedError

    def fit_transform(
        self,
        df: pd.DataFrame,
        *args: Any,
        **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tiện ích gộp fit() + transform() cho dữ liệu huấn luyện.

        Args:
            df (pd.DataFrame): DataFrame huấn luyện

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) sau khi tiền xử lý
        """
        self.fit(df)
        X, y = self.transform(df, *args, **kwargs)
        if y is None:
            raise RuntimeError("fit_transform() kỳ vọng data có chứa target, nhưng transform() trả về y=None.")
        return X, y

    def get_feature_names(self) -> List[str]:
        """
        Trả về danh sách tên features sau preprocessing.

        Returns:
            List[str]: danh sách tên cột (features)
        """
        return self.feature_names

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"target_col={self.target_col}, "
            f"status={status}, "
            f"n_features={len(self.feature_names)}"
            f")"
        )
