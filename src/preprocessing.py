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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Class để tiền xử lý dữ liệu phim cho mô hình dự đoán revenue.
    
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
        feature_names (List[str]): Danh sách tên features sau preprocessing
        is_fitted (bool): Trạng thái đã fit hay chưa
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
        
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        
        logger.info("DataPreprocessor đã được khởi tạo thành công")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý missing values cho tất cả các cột.
        
        Args:
            df (pd.DataFrame): DataFrame cần xử lý
            
        Returns:
            pd.DataFrame: DataFrame đã xử lý missing values
        """
        df = df.copy()
        
        # Xử lý numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_strategy = self.preprocess_config['numeric_imputation']
        
        for col in numeric_cols:
            if df[col].isnull().any():
                if numeric_strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif numeric_strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif numeric_strategy == 'zero':
                    df[col].fillna(0, inplace=True)
        
        # Xử lý categorical/text columns
        text_cols = ['overview', 'genres', 'production_companies']
        for col in text_cols:
            if col in df.columns:
                df[col].fillna('', inplace=True)
        
        # Xử lý release_date
        if 'release_date' in df.columns:
            df['release_date'].fillna(method='ffill', inplace=True)
        
        logger.info("Đã xử lý missing values")
        return df
    
    def _detect_and_remove_outliers(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Phát hiện và loại bỏ outliers sử dụng IQR method.
        
        Args:
            df (pd.DataFrame): DataFrame cần xử lý
            columns (List[str], optional): Danh sách cột cần kiểm tra outliers.
                Nếu None, sử dụng config.
                
        Returns:
            pd.DataFrame: DataFrame đã loại bỏ outliers
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
        logger.info(f"Đã loại bỏ {removed_count} outliers ({removed_count/initial_shape*100:.2f}%)")
        
        return df
    
    def _engineer_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo features từ cột release_date.
        
        Args:
            df (pd.DataFrame): DataFrame chứa cột release_date
            
        Returns:
            pd.DataFrame: DataFrame với các date features mới
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
        
        # Drop original release_date
        df = df.drop(columns=['release_date'])
        
        logger.info(f"Đã tạo {len(date_features)} date features")
        return df
    
    def _process_genres(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Xử lý cột genres bằng MultiLabelBinarizer.
        
        Args:
            df (pd.DataFrame): DataFrame chứa cột genres
            fit (bool): Có fit MultiLabelBinarizer hay không
            
        Returns:
            pd.DataFrame: DataFrame với genres đã được one-hot encoded
        """
        df = df.copy()
        
        if 'genres' not in df.columns:
            logger.warning("Không tìm thấy cột genres")
            return df
        
        # Chuyển genres string thành list
        # Ví dụ: "Action|Adventure" -> ["Action", "Adventure"]
        genres_lists = df['genres'].apply(lambda x: x.split('|') if x else [])
        
        if fit:
            # Fit và transform
            genres_encoded = self.mlb_genres.fit_transform(genres_lists)
        else:
            # Chỉ transform
            genres_encoded = self.mlb_genres.transform(genres_lists)
        
        # Tạo DataFrame từ encoded genres
        genres_df = pd.DataFrame(
            genres_encoded,
            columns=[f'genre_{genre}' for genre in self.mlb_genres.classes_],
            index=df.index
        )
        
        # Gộp vào df chính và drop cột genres gốc
        df = pd.concat([df, genres_df], axis=1)
        df = df.drop(columns=['genres'])
        
        logger.info(f"Đã encode {len(self.mlb_genres.classes_)} genres")
        return df
    
    def _process_overview_tfidf(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Xử lý cột overview bằng TF-IDF vectorization.
        
        Args:
            df (pd.DataFrame): DataFrame chứa cột overview
            fit (bool): Có fit TfidfVectorizer hay không
            
        Returns:
            pd.DataFrame: DataFrame với TF-IDF features từ overview
        """
        df = df.copy()
        
        if 'overview' not in df.columns:
            logger.warning("Không tìm thấy cột overview")
            return df
        
        # Fill missing overview với empty string
        overview_texts = df['overview'].fillna('')
        
        if fit:
            # Fit và transform
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(overview_texts)
        else:
            # Chỉ transform
            tfidf_matrix = self.tfidf_vectorizer.transform(overview_texts)
        
        # Tạo DataFrame từ TF-IDF matrix
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
            index=df.index
        )
        
        # Gộp vào df chính và drop cột overview gốc
        df = pd.concat([df, tfidf_df], axis=1)
        df = df.drop(columns=['overview'])
        
        logger.info(f"Đã tạo {tfidf_matrix.shape[1]} TF-IDF features từ overview")
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo các features phái sinh từ các cột hiện có.
        
        Args:
            df (pd.DataFrame): DataFrame đầu vào
            
        Returns:
            pd.DataFrame: DataFrame với các derived features
        """
        df = df.copy()
        
        # Budget per minute (nếu có runtime)
        if 'budget' in df.columns and 'runtime' in df.columns:
            df['budget_per_minute'] = df['budget'] / (df['runtime'] + 1)  # +1 để tránh chia 0
        
        # Vote score (weighted rating)
        if 'vote_average' in df.columns and 'vote_count' in df.columns:
            df['vote_score'] = df['vote_average'] * np.log1p(df['vote_count'])
        
        # Popularity per vote
        if 'popularity' in df.columns and 'vote_count' in df.columns:
            df['popularity_per_vote'] = df['popularity'] / (df['vote_count'] + 1)
        
        # Log transformations (để giảm skewness)
        for col in ['budget', 'popularity', 'vote_count']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col])
        
        logger.info("Đã tạo derived features")
        return df
    
    def _select_features_for_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chọn các features cần thiết cho modeling và drop các cột không cần.
        
        Args:
            df (pd.DataFrame): DataFrame đầy đủ
            
        Returns:
            pd.DataFrame: DataFrame chỉ chứa features cần thiết
        """
        df = df.copy()
        
        # Các cột cần drop (không phải features)
        cols_to_drop = [
            'id', 'title', 'original_language', 'status',
            'production_companies', 'revenue'  # revenue là target
        ]
        
        # Drop các cột tồn tại
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        return df
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessor trên training data.
        
        Phương thức này sẽ học các tham số cần thiết từ training data
        như scaler parameters, genre classes, TF-IDF vocabulary.
        
        Args:
            df (pd.DataFrame): Training DataFrame (bao gồm cả target 'revenue')
            
        Returns:
            DataPreprocessor: Self để có thể chain methods
            
        Raises:
            ValueError: Nếu không tìm thấy cột revenue
        """
        if 'revenue' not in df.columns:
            raise ValueError("DataFrame phải có cột 'revenue' (target)")
        
        logger.info("Bắt đầu fit preprocessing pipeline...")
        
        df = df.copy()
        
        # 1. Handle missing values
        df = self._handle_missing_values(df)
        
        # 2. Remove outliers (chỉ fit, không áp dụng lên test set)
        df = self._detect_and_remove_outliers(df)
        
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
        
        Args:
            df (pd.DataFrame): DataFrame cần transform
            remove_outliers (bool): Có loại bỏ outliers hay không (mặc định False cho test set)
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: 
                - X: Features đã được transform
                - y: Target (revenue) nếu có trong df, None nếu không có
                
        Raises:
            RuntimeError: Nếu preprocessor chưa được fit
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor chưa được fit. Vui lòng gọi fit() trước.")
        
        logger.info("Bắt đầu transform data...")
        
        df = df.copy()
        
        # Lưu target nếu có
        y = None
        if 'revenue' in df.columns:
            y = df['revenue'].values
        
        # 1. Handle missing values
        df = self._handle_missing_values(df)
        
        # 2. Remove outliers (optional)
        if remove_outliers:
            df = self._detect_and_remove_outliers(df)
            if y is not None:
                y = y[df.index]  # Đồng bộ y với df sau khi loại outliers
        
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
        
        logger.info(f"Đã transform data với shape: {X.shape}")
        
        return X, y
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit và transform trong một bước (cho training data).
        
        Args:
            df (pd.DataFrame): Training DataFrame (bao gồm cả target 'revenue')
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (features) và y (target)
        """
        self.fit(df)
        X, y = self.transform(df, remove_outliers=True)
        return X, y
    
    def save_processed_data(
        self, 
        X: np.ndarray, 
        filepath: str,
        y: Optional[np.ndarray] = None
    ) -> None:
        """
        Lưu dữ liệu đã xử lý vào file CSV.
        
        Args:
            X (np.ndarray): Features array
            filepath (str): Đường dẫn file để lưu
            y (np.ndarray, optional): Target array (sẽ được thêm vào cột cuối)
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Tạo DataFrame từ numpy array
        df = pd.DataFrame(X, columns=self.feature_names)
        
        if y is not None:
            df['revenue'] = y
        
        df.to_csv(filepath, index=False)
        logger.info(f"Đã lưu processed data vào: {filepath}")
    
    def save_preprocessor(self, filepath: str) -> None:
        """
        Lưu toàn bộ preprocessor object vào file.
        
        Args:
            filepath (str): Đường dẫn file để lưu (.pkl)
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, filepath)
        logger.info(f"Đã lưu preprocessor vào: {filepath}")
    
    @staticmethod
    def load_preprocessor(filepath: str) -> 'DataPreprocessor':
        """
        Load preprocessor từ file.
        
        Args:
            filepath (str): Đường dẫn file preprocessor
            
        Returns:
            DataPreprocessor: Preprocessor đã được load
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
        
        preprocessor = joblib.load(filepath)
        logger.info(f"Đã load preprocessor từ: {filepath}")
        
        return preprocessor
    
    def get_feature_names(self) -> List[str]:
        """
        Lấy danh sách tên features sau preprocessing.
        
        Returns:
            List[str]: Danh sách feature names
        """
        return self.feature_names
    
    def __repr__(self) -> str:
        """String representation của DataPreprocessor."""
        status = "fitted" if self.is_fitted else "not fitted"
        n_features = len(self.feature_names) if self.is_fitted else 0
        return (
            f"DataPreprocessor("
            f"status={status}, "
            f"n_features={n_features}, "
            f"scaler={self.scaler.__class__.__name__}"
            f")"
        )


if __name__ == "__main__":
    from data_loader import TMDbDataLoader
    
    # Load dữ liệu
    df = TMDbDataLoader.load_from_csv("data/raw/movies_2020_2024.csv")
    
    # Khởi tạo preprocessor
    preprocessor = DataPreprocessor()
    
    # Fit và transform
    X, y = preprocessor.fit_transform(df)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature names: {preprocessor.get_feature_names()[:10]}...")
    
    # Lưu processed data
    preprocessor.save_processed_data(X, "data/processed/X_train.csv", y)
    
    # Lưu preprocessor
    preprocessor.save_preprocessor("models/preprocessor.pkl")