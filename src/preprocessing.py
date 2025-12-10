"""
Module Data Preprocessing

Module chứa class DataPreprocessor để thực hiện toàn bộ quy trình
tiền xử lý dữ liệu bao gồm: xử lý missing values, outlier detection,
feature engineering, encoding, và scaling.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import yaml
import logging
import joblib
import os

from src.base_preprocessing import BasePreprocessor  
# Setup logging - use parent logger configuration
logger = logging.getLogger(__name__)


class DataPreprocessor(BasePreprocessor):
    """
    Class để tiền xử lý dữ liệu phim cho mô hình dự đoán revenue.

    Kế thừa BasePreprocessor để:
    - Chuẩn hóa giao diện fit / transform / fit_transform
    - Dễ tái sử dụng / thay thế bằng preprocessor khác trong tương lai

    Class thực hiện toàn bộ pipeline tiền xử lý bao gồm:
    - Xử lý missing values với nhiều chiến lược
    - Phát hiện và xử lý outliers bằng IQR method
    - Feature engineering từ date, genres, và text
    - TF-IDF vectorization cho overview
    - Encoding categorical variables
    - Scaling numerical features

    Attributes:
        config (dict): Configuration từ file config.yaml
        scaler: Scaler object (StandardScaler hoặc MinMaxScaler)
        mlb_genres: MultiLabelBinarizer cho genres
        tfidf_vectorizer: TfidfVectorizer cho overview
        feature_names (List[str]): Danh sách tên features sau preprocessing (kế thừa từ BasePreprocessor)
        is_fitted (bool): Trạng thái đã fit hay chưa (kế thừa từ BasePreprocessor)
        target_col (str): Tên cột target, mặc định 'revenue' (kế thừa từ BasePreprocessor)
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Khởi tạo DataPreprocessor với configuration.

        Args:
            config_path (str): Đường dẫn đến file config.yaml

        Raises:
            FileNotFoundError: Nếu không tìm thấy file config
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.preprocess_config = self.config['preprocessing']

        # Gọi BasePreprocessor __init__ sau khi có config
        # Cho phép override tên cột target trong config nếu muốn tái sử dụng cho bài toán khác
        target_col = self.preprocess_config.get('target_col', 'revenue')
        super().__init__(name="movie_preprocessor", target_col=target_col)

        # Khởi tạo các transformers
        scaling_method = self.preprocess_config['scaling_method']
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Scaling method không hợp lệ: {scaling_method}")

        self.mlb_genres = MultiLabelBinarizer()

        tfidf_config = self.preprocess_config['tfidf']
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_config['max_features'],
            min_df=tfidf_config['min_df'],
            max_df=tfidf_config['max_df'],
            ngram_range=tuple(tfidf_config['ngram_range']),
            stop_words='english'
        )

        logger.info(
            f"DataPreprocessor đã được khởi tạo thành công "
            f"(target_col={self.target_col}, scaler={self.scaler.__class__.__name__})"
        )

    @staticmethod
    def read_data(filepath: str) -> pd.DataFrame:
        """
        Đọc dữ liệu từ file (hỗ trợ csv, xlsx, json).

        Args:
            filepath (str): Đường dẫn đến file dữ liệu

        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu

        Raises:
            ValueError: Nếu định dạng file không được hỗ trợ
        """
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            return pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        else:
            raise ValueError(
                f"Định dạng file không hỗ trợ: {filepath}. "
                f"Chỉ hỗ trợ .csv, .xlsx, .json"
            )

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý missing values cho tất cả các cột.
        """
        df = df.copy()

        # Xử lý numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_strategy = self.preprocess_config['numeric_imputation']

        for col in numeric_cols:
            if df[col].isnull().any():
                if numeric_strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif numeric_strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif numeric_strategy == 'zero':
                    df[col] = df[col].fillna(0)

        # Xử lý categorical/text columns
        text_cols = self.preprocess_config.get('text_columns', [])
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')

        # Xử lý explicitly cho genres và overview (critical for downstream processing)
        if 'genres' in df.columns:
            df['genres'] = df['genres'].fillna('')
        if 'overview' in df.columns:
            df['overview'] = df['overview'].fillna('')

        # Xử lý release_date
        if 'release_date' in df.columns:
            df['release_date'] = df['release_date'].ffill()

        logger.info("Đã xử lý missing values")
        return df

    def _detect_and_remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Phát hiện và loại bỏ outliers sử dụng IQR method.
        """
        if not self.preprocess_config['outlier_detection']['enabled']:
            logger.info("Outlier detection bị tắt trong config")
            return df

        df = df.copy()

        if columns is None:
            columns = self.preprocess_config['outlier_detection']['columns']

        # Chỉ xử lý các cột tồn tại trong df
        columns = [col for col in columns if col in df.columns]

        iqr_multiplier = self.preprocess_config['outlier_detection']['iqr_multiplier']
        initial_shape = df.shape[0]

        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR

            # Lọc bỏ outliers
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        removed_count = initial_shape - df.shape[0]
        if initial_shape > 0:
            logger.info(
                f"Đã loại bỏ {removed_count} outliers "
                f"({removed_count / initial_shape * 100:.2f}%)"
            )
        else:
            logger.info("Không có dữ liệu để kiểm tra outliers")

        return df

    def _engineer_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo features từ cột release_date.
        """
        df = df.copy()

        if 'release_date' not in df.columns:
            logger.warning("Không tìm thấy cột release_date")
            return df

        # Convert to datetime
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

        date_features = self.preprocess_config['date_features']

        if 'year' in date_features:
            df['release_year'] = df['release_date'].dt.year

        if 'month' in date_features:
            df['release_month'] = df['release_date'].dt.month

        if 'quarter' in date_features:
            df['release_quarter'] = df['release_date'].dt.quarter

        if 'day_of_week' in date_features:
            df['release_day_of_week'] = df['release_date'].dt.dayofweek

        if 'is_weekend' in date_features:
            df['is_weekend'] = df['release_date'].dt.dayofweek.isin([5, 6]).astype(int)

        df = df.drop(columns=['release_date'])

        logger.info(f"Đã tạo {len(date_features)} date features")
        return df

    def _process_genres(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Xử lý cột genres bằng MultiLabelBinarizer.
        """
        df = df.copy()

        if 'genres' not in df.columns:
            logger.warning("Không tìm thấy cột genres")
            return df

        # Fix: Handle NaN values properly - check if string type AND not empty
        genres_lists = df['genres'].apply(
            lambda x: x.split('|') if isinstance(x, str) and x else []
        )

        if fit:
            genres_encoded = self.mlb_genres.fit_transform(genres_lists)
        else:
            genres_encoded = self.mlb_genres.transform(genres_lists)

        genres_df = pd.DataFrame(
            genres_encoded,
            columns=[f'genre_{genre}' for genre in self.mlb_genres.classes_],
            index=df.index
        )

        df = pd.concat([df, genres_df], axis=1)
        df = df.drop(columns=['genres'])

        logger.info(f"Đã encode {len(self.mlb_genres.classes_)} genres")
        return df

    def _process_overview_tfidf(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Xử lý cột overview bằng TF-IDF vectorization.
        """
        df = df.copy()

        if 'overview' not in df.columns:
            logger.warning("Không tìm thấy cột overview")
            return df

        overview_texts = df['overview'].fillna('')

        if fit:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(overview_texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(overview_texts)

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
            index=df.index
        )

        df = pd.concat([df, tfidf_df], axis=1)
        df = df.drop(columns=['overview'])

        logger.info(f"Đã tạo {tfidf_matrix.shape[1]} TF-IDF features từ overview")
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo các features phái sinh từ các cột hiện có.
        """
        df = df.copy()

        # Budget per minute
        if 'budget' in df.columns and 'runtime' in df.columns:
            df['budget_per_minute'] = df['budget'] / (df['runtime'] + 1)

        # Vote score (weighted rating)
        if 'vote_average' in df.columns and 'vote_count' in df.columns:
            df['vote_score'] = df['vote_average'] * np.log1p(df['vote_count'])

        # Popularity per vote
        if 'popularity' in df.columns and 'vote_count' in df.columns:
            df['popularity_per_vote'] = df['popularity'] / (df['vote_count'] + 1)

        # Log transformations
        for col in ['budget', 'popularity', 'vote_count']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col])

        logger.info("Đã tạo derived features")
        return df

    def _select_features_for_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chọn các features cần thiết cho modeling và drop các cột không cần.
        """
        df = df.copy()

        cols_to_drop = [
            'id', 'title', 'original_language', 'status',
            'production_companies', self.target_col  # target
        ]

        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)

        return df

    def _prepare_data(self, df: pd.DataFrame, extract_target: bool = True):
        """Helper method để chuẩn hóa cách xử lý target."""
        df = df.copy()
        y_series = None
        
        if extract_target and self.target_col in df.columns:
            y_series = df[self.target_col].copy()
            df = df.drop(columns=[self.target_col])
        
        return df, y_series

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit preprocessor trên training data.

        Học:
        - tham số xử lý missing
        - tập outliers threshold
        - classes cho genres
        - vocabulary TF-IDF
        - scaler parameters
        """
        if self.target_col not in df.columns:
            raise ValueError(f"DataFrame phải có cột '{self.target_col}' (target)")

        logger.info("Bắt đầu fit preprocessing pipeline...")

        df = df.copy()
        # Extract target trước để outlier detection chỉ dựa vào X
        df, y_series = self._prepare_data(df, extract_target=False)

        # 1. Handle missing values
        df = self._handle_missing_values(df)

        # 2. Remove outliers (chỉ fit trên train)
        df = self._detect_and_remove_outliers(df)

        # Đồng bộ y với df sau khi loại outliers (giả sử index là RangeIndex)
        if y_series is not None:
            y_series = y_series.loc[df.index]

        # 3. Engineer date features
        df = self._engineer_date_features(df)

        # 4. Process genres (fit=True)
        df = self._process_genres(df, fit=True)

        # 5. Process overview with TF-IDF (fit=True)
        df = self._process_overview_tfidf(df, fit=True)

        # 6. Add derived features
        df = self._add_derived_features(df)

        # 7. Select features
        df = self._select_features_for_modeling(df)

        # 8. Fit scaler trên numerical features
        self.scaler.fit(df)

        # Lưu feature names
        self.feature_names = df.columns.tolist()
        self.is_fitted = True

        logger.info(f"Đã fit preprocessor với {len(self.feature_names)} features")
        return self

    def transform(
        self,
        df: pd.DataFrame,
        remove_outliers: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data sử dụng fitted preprocessor.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor chưa được fit. Vui lòng gọi fit() trước.")

        logger.info("Bắt đầu transform data...")

        df = df.copy()

        # Lưu target nếu có
        y = None
        if self.target_col in df.columns:
            y_series = df[self.target_col].copy()  # Giữ index
            df = df.drop(columns=[self.target_col])

        # 1. Handle missing values
        df = self._handle_missing_values(df)

        # 2. Remove outliers (optional)
        if remove_outliers:
            original_index = df.index
            df = self._detect_and_remove_outliers(df)
            if y_series is not None:
                # Đồng bộ y với df sau khi loại outliers (giả sử index là RangeIndex)
                y_series = y_series.loc[df.index]

        # 3. Engineer date features
        df = self._engineer_date_features(df)

        # 4. Process genres (fit=False)
        df = self._process_genres(df, fit=False)

        # 5. Process overview with TF-IDF (fit=False)
        df = self._process_overview_tfidf(df, fit=False)

        # 6. Add derived features
        df = self._add_derived_features(df)

        # 7. Select features
        df = self._select_features_for_modeling(df)

        # 8. Transform với scaler
        X = self.scaler.transform(df)
        if y_series is not None:
            y = np.log1p(y_series.values)

        logger.info(f"Đã transform data với shape: {X.shape}")
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit và transform trong một bước (cho training data).
        (Override nhẹ để giữ đúng kiểu trả về & sử dụng remove_outliers=True cho train)
        """
        # dùng BasePreprocessor.fit_transform logic
        self.fit(df)
        X, y = self.transform(df, remove_outliers=True)
        if y is None:
            raise RuntimeError(
                "fit_transform() kỳ vọng data có chứa target, "
                f"nhưng không tìm thấy cột '{self.target_col}'."
            )
        return X, y

    def save_processed_data(
        self,
        X: np.ndarray,
        filepath: str,
        y: Optional[np.ndarray] = None
    ) -> None:
        """
        Lưu dữ liệu đã xử lý vào file CSV.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(X, columns=self.feature_names)

        if y is not None:
            df[self.target_col] = y

        df.to_csv(filepath, index=False)
        logger.info(f"Đã lưu processed data vào: {filepath}")

    def save_preprocessor(self, filepath: str) -> None:
        """
        Lưu toàn bộ preprocessor object vào file.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Đã lưu preprocessor vào: {filepath}")

    @staticmethod
    def load_preprocessor(filepath: str) -> "DataPreprocessor":
        """
        Load preprocessor từ file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

        preprocessor = joblib.load(filepath)
        logger.info(f"Đã load preprocessor từ: {filepath}")
        return preprocessor

    def inverse_transform_target(self, y_log: np.ndarray) -> np.ndarray:
        """
        Inverse transform target từ log scale về original scale.
        """
        return np.expm1(y_log)

    # get_feature_names() đã có sẵn từ BasePreprocessor,
    # nhưng vẫn giữ lại cho rõ ràng nếu bạn muốn dùng trực tiếp.
    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        n_features = len(self.feature_names) if self.is_fitted else 0
        return (
            f"DataPreprocessor("
            f"name={self.name}, "
            f"status={status}, "
            f"n_features={n_features}, "
            f"target_col={self.target_col}, "
            f"scaler={self.scaler.__class__.__name__}"
            f")"
        )


if __name__ == "__main__":
    from src.data_loader import TMDbDataLoader

    df = TMDbDataLoader.load_from_csv("data/raw/movies_2020_2024.csv")

    preprocessor = DataPreprocessor()

    X, y = preprocessor.fit_transform(df)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature names: {preprocessor.get_feature_names()[:10]}...")

    preprocessor.save_processed_data(X, "data/processed/X_train.csv", y)

    preprocessor.save_preprocessor("models/preprocessor.pkl")
