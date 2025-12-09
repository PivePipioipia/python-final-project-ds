"""
Module Data Preprocessing V2 (Advanced)

Phiên bản nâng cao của DataPreprocessor, áp dụng các kỹ thuật:
- KNN Imputation thay vì Simple Imputation
- RobustScaler thay vì StandardScaler
- BAAI/bge-small-en-v1.5 Embeddings thay vì TF-IDF (SOTA Semantic Search)
- Enhanced Date Features (Seasonality)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.preprocessing import RobustScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import KNNImputer
import yaml
import logging
import os
import joblib
from pathlib import Path

# Thử import sentence_transformer, nếu chưa cài thì fallback hoặc báo lỗi
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from src.base_preprocessing import BasePreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessorV2(BasePreprocessor):
    """
    Class tiền xử lý dữ liệu nâng cao (V2).
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.preprocess_config = self.config['preprocessing']
        target_col = self.preprocess_config.get('target_col', 'revenue')
        super().__init__(name="movie_preprocessor_v2", target_col=target_col)

        # 1. Advanced Scaling: RobustScaler (Handle Outliers)
        # Giữ lại outliers nhưng scale chúng bằng IQR để mô hình không bị nhiễu
        self.scaler = RobustScaler()
        
        # 2. Advanced Imputation: KNNImputer (Missing Values)
        # Tìm 5 phim tương đồng (k-NN) để điền dữ liệu khuyết
        self.imputer = KNNImputer(n_neighbors=5)

        self.mlb_genres = MultiLabelBinarizer()
        
        # 3. NLP Embedding: BGE-Small (Rich Text Features)
        # Thay thế TF-IDF bằng Deep Learning Embedding 
        if HAS_SENTENCE_TRANSFORMERS:
            self.embedding_model_name = 'BAAI/bge-small-en-v1.5'
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Using Embedding Model: {self.embedding_model_name}")
        else:
            logger.warning("Thư viện 'sentence-transformers' chưa được cài đặt. Fallback về TF-IDF.")
            tfidf_config = self.preprocess_config['tfidf']
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=tfidf_config['max_features'],
                min_df=tfidf_config['min_df'],
                max_df=tfidf_config['max_df'],
                ngram_range=tuple(tfidf_config['ngram_range']),
                stop_words='english'
            )

        logger.info("DataPreprocessorV2 (Advanced) đã được khởi tạo.")

    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Kỹ thuật: KNN Imputation (Thông minh hơn điền Mean)"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            if fit:
                df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
            else:
                df[numeric_cols] = self.imputer.transform(df[numeric_cols])

        text_cols = ['overview', 'genres', 'production_companies']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')

        if 'release_date' in df.columns:
            df['release_date'] = df['release_date'].ffill()

        return df

    def _detect_and_remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kỹ thuật: Keep All Data (Quan trọng cho mô hình học được 'Blockbuster')"""
        # Không xóa dòng nào cả. RobustScaler sẽ lo phần scaling.
        return df

    def _engineer_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kỹ thuật: Seasonality (Từ EDA Seasonality)"""
        df = df.copy()
        if 'release_date' not in df.columns: return df

        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        df['release_day_of_week'] = df['release_date'].dt.dayofweek
        
        # New: Mùa hè (Blockbuster season) & Mùa lễ (Holiday season)
        df['is_summer_season'] = df['release_month'].isin([5, 6, 7]).astype(int)
        df['is_holiday_season'] = df['release_month'].isin([11, 12]).astype(int)
        
        # New: Đầu quý / Cuối quý (Thời điểm chốt sổ tài chính)
        df['is_quarter_start'] = df['release_date'].dt.is_quarter_start.astype(int)

        df = df.drop(columns=['release_date'])
        return df

    def _process_genres(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        if 'genres' not in df.columns: return df
        
        genres_lists = df['genres'].apply(lambda x: x.split('|') if x else [])
        
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
        return df

    def _process_overview(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Kỹ thuật: Semantic Embeddings (Hiểu ngữ nghĩa nội dung)"""
        df = df.copy()
        if 'overview' not in df.columns: return df
        
        overview_texts = df['overview'].fillna('').tolist()
        
        if HAS_SENTENCE_TRANSFORMERS:
            # Embedding không cần 'fit' theo nghĩa thống kê (nó là pre-trained model)
            # Nhưng để thống nhất interface, ta cứ gọi encode
            logger.info("Encoding overview with BGE Embeddings...")
            embeddings = self.embedding_model.encode(overview_texts, show_progress_bar=False)
            
            # Embeddings shape: (n_samples, 384)
            feature_names = [f'embed_{i}' for i in range(embeddings.shape[1])]
            
            embed_df = pd.DataFrame(
                embeddings,
                columns=feature_names,
                index=df.index
            )
            df = pd.concat([df, embed_df], axis=1)
            logger.info(f"Đã tạo {embeddings.shape[1]} features từ BGE Embeddings")
            
        else:
            # Fallback TF-IDF
            if fit:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(overview_texts)
            else:
                tfidf_matrix = self.tfidf_vectorizer.transform(overview_texts)
            
            cols = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=cols, index=df.index)
            df = pd.concat([df, tfidf_df], axis=1)
            
        df = df.drop(columns=['overview'])
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kỹ thuật: Robust Division & Log Transform"""
        df = df.copy()
        
        if 'budget' in df.columns and 'runtime' in df.columns:
            # Runtime 0 -> 90 mins (Logic an toàn)
            safe_runtime = df['runtime'].replace(0, 90)
            df['budget_per_minute'] = df['budget'] / safe_runtime

        if 'vote_average' in df.columns and 'vote_count' in df.columns:
            df['vote_score'] = df['vote_average'] * np.log1p(df['vote_count'])

        for col in ['budget', 'popularity', 'vote_count']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col])

        return df

    def _select_features_for_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols_to_drop = ['id', 'title', 'original_language', 'status', 
                        'production_companies', self.target_col]
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        return df

    def fit(self, df: pd.DataFrame) -> "DataPreprocessorV2":
        if self.target_col not in df.columns:
            raise ValueError(f"Missing target column: {self.target_col}")

        logger.info("Bắt đầu fit DataPreprocessorV2 (Advanced)...")
        df = df.copy()

        # Pipeline:
        df = self._handle_missing_values(df, fit=True)
        # Outliers: Skiped
        df = self._engineer_date_features(df)
        df = self._process_genres(df, fit=True)
        df = self._process_overview(df, fit=True) # Now uses Embeddings
        df = self._add_derived_features(df)
        df = self._select_features_for_modeling(df)
        
        self.scaler.fit(df)
        
        self.feature_names = df.columns.tolist()
        self.is_fitted = True
        logger.info(f"Đã fit V2 với {len(self.feature_names)} features.")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.is_fitted:
            raise RuntimeError("Chưa fit preprocessor!")

        df = df.copy()
        y = None
        if self.target_col in df.columns:
            y = df[self.target_col].values

        df = self._handle_missing_values(df, fit=False)
        # Outliers: Skiped
        df = self._engineer_date_features(df)
        df = self._process_genres(df, fit=False)
        df = self._process_overview(df, fit=False)
        df = self._add_derived_features(df)
        df = self._select_features_for_modeling(df)
        
        if self.scaler:
             # Ensure columns match exactly what scaler was fitted on
            df = df.reindex(columns=self.feature_names, fill_value=0)
            
        X = self.scaler.transform(df)
        return X, y

    def save_processed_data(self, X: np.ndarray, filepath: str, y: Optional[np.ndarray] = None) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(X, columns=self.feature_names)
        if y is not None:
            df[self.target_col] = y
        df.to_csv(filepath, index=False)
        logger.info(f"Đã lưu processed data vào: {filepath}")

    def save_preprocessor(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Đã lưu preprocessor vào: {filepath}")

    @staticmethod
    def load_preprocessor(filepath: str) -> "DataPreprocessorV2":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
        preprocessor = joblib.load(filepath)
        return preprocessor
