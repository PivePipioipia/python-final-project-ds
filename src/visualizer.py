"""
Module Visualizer

Module chứa class Visualizer để trực quan hóa dữ liệu và kết quả mô hình.
Bao gồm EDA plots, feature importance, model comparison, và prediction analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import yaml
import logging
import os

# Setup matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Visualizer:
    """
    Class để trực quan hóa dữ liệu và kết quả mô hình.
    
    Class cung cấp các phương thức để:
    - Vẽ biểu đồ phân tích dữ liệu khám phá (EDA)
    - Hiển thị feature importance của models
    - So sánh performance giữa các models
    - Phân tích predictions (actual vs predicted, residuals)
    - Lưu plots vào files với chất lượng cao
    
    Attributes:
        config (dict): Configuration từ file config.yaml
        viz_config (dict): Visualization configuration
        save_dir (str): Thư mục để lưu plots
    """
    
    def __init__(
        self, 
        config_path: str = "configs/config.yaml",
        save_dir: str = "visualizations"
    ):
        """
        Khởi tạo Visualizer với configuration.
        
        Args:
            config_path (str): Đường dẫn đến file config.yaml
            save_dir (str): Thư mục để lưu plots
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.viz_config = self.config['visualization']
        self.save_dir = save_dir
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{save_dir}/eda_plots").mkdir(parents=True, exist_ok=True)
        Path(f"{save_dir}/model_results").mkdir(parents=True, exist_ok=True)
        
        self.figsize = tuple(self.viz_config['figure_size'])
        self.dpi = self.viz_config['dpi']
        
        logger.info("Visualizer đã được khởi tạo thành công")
    
    def plot_target_distribution(
        self, 
        y: np.ndarray,
        title: str = "Phân phối Revenue",
        log_scale: bool = True
    ) -> None:
        """
        Vẽ phân phối của target variable (revenue).
        
        Args:
            y (np.ndarray): Target values
            title (str): Tiêu đề biểu đồ
            log_scale (bool): Có dùng log scale hay không (vì revenue thường skewed)
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        axes[0].hist(y, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Revenue (USD)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{title} - Histogram')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].boxplot(y, vert=True)
        axes[1].set_ylabel('Revenue (USD)')
        axes[1].set_title(f'{title} - Boxplot')
        axes[1].grid(True, alpha=0.3)
        
        if log_scale:
            axes[0].set_yscale('log')
            axes[1].set_yscale('log')
        
        plt.tight_layout()
        logger.info(f"Đã vẽ biểu đồ phân phối target: {title}")
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        title: str = "Ma trận Tương quan",
        top_n: int = 20
    ) -> None:
        """
        Vẽ correlation matrix cho top N features.
        
        Args:
            df (pd.DataFrame): DataFrame chứa features
            title (str): Tiêu đề biểu đồ
            top_n (int): Số lượng features để hiển thị
        """
        if 'revenue' in df.columns:
            correlations = df.corr()['revenue'].abs().sort_values(ascending=False)
            top_features = correlations.head(top_n).index.tolist()
            df_subset = df[top_features]
        else:
            df_subset = df.iloc[:, :top_n]
        
        corr_matrix = df_subset.corr()
        
        plt.figure(figsize=(self.figsize[0], self.figsize[0]))
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        logger.info(f"Đã vẽ correlation matrix: {title}")
    
    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        features: List[str],
        title: str = "Phân phối Features"
    ) -> None:
        """
        Vẽ phân phối của nhiều features.
        
        Args:
            df (pd.DataFrame): DataFrame chứa features
            features (List[str]): Danh sách features cần vẽ
            title (str): Tiêu đề chung
        """
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            if feature not in df.columns:
                continue
            
            ax = axes[idx]
            df[feature].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {feature}')
            ax.grid(True, alpha=0.3)
        
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        logger.info(f"Đã vẽ phân phối cho {n_features} features")
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        model_name: str,
        top_n: Optional[int] = None
    ) -> None:
        """
        Vẽ biểu đồ feature importance.
        
        Args:
            importance_df (pd.DataFrame): DataFrame chứa feature và importance
            model_name (str): Tên model
            top_n (int, optional): Số lượng top features. Mặc định từ config.
        """
        if top_n is None:
            top_n = self.viz_config['top_n_features']
        
        importance_df = importance_df.head(top_n).copy()
        
        plt.figure(figsize=self.figsize)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        
        plt.barh(
            importance_df['feature'],
            importance_df['importance'],
            color=colors,
            edgecolor='black'
        )
        
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(
            f'Top {top_n} Feature Importance - {model_name.upper()}',
            fontsize=14,
            fontweight='bold'
        )
        plt.gca().invert_yaxis()  
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        logger.info(f"Đã vẽ feature importance cho {model_name}")
    
    def plot_actual_vs_predicted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> None:
        """
        Vẽ biểu đồ scatter actual vs predicted values.
        
        Args:
            y_true (np.ndarray): Giá trị thực tế
            y_pred (np.ndarray): Giá trị dự đoán
            model_name (str): Tên model
        """
        plt.figure(figsize=self.figsize)
        
        plt.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Revenue (USD)', fontsize=12)
        plt.ylabel('Predicted Revenue (USD)', fontsize=12)
        plt.title(
            f'Actual vs Predicted Revenue - {model_name.upper()}',
            fontsize=14,
            fontweight='bold'
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        logger.info(f"Đã vẽ actual vs predicted cho {model_name}")
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> None:
        """
        Vẽ biểu đồ residual analysis.
        
        Args:
            y_true (np.ndarray): Giá trị thực tế
            y_pred (np.ndarray): Giá trị dự đoán
            model_name (str): Tên model
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Revenue (USD)', fontsize=11)
        axes[0].set_ylabel('Residuals (USD)', fontsize=11)
        axes[0].set_title('Residual Plot', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residuals (USD)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(
            f'Residual Analysis - {model_name.upper()}',
            fontsize=14,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()
        
        logger.info(f"Đã vẽ residual analysis cho {model_name}")
    
    def plot_model_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str = 'RMSE'
    ) -> None:
        """
        Vẽ biểu đồ so sánh performance giữa các models.
        
        Args:
            results_df (pd.DataFrame): DataFrame chứa kết quả của các models
            metric (str): Metric để so sánh ('RMSE', 'MAE', 'R2', 'MAPE')
        """
        if metric not in results_df.columns:
            raise ValueError(f"Metric {metric} không tồn tại trong results_df")
        
        ascending = metric != 'R2'
        results_sorted = results_df.sort_values(metric, ascending=ascending)
        
        plt.figure(figsize=self.figsize)
        
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_sorted))]
        
        bars = plt.barh(
            results_sorted.index,
            results_sorted[metric],
            color=colors,
            edgecolor='black',
            linewidth=1.5
        )
        
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f' {width:,.2f}',
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        plt.xlabel(metric, fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title(
            f'Model Comparison - {metric}',
            fontsize=14,
            fontweight='bold'
        )
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        logger.info(f"Đã vẽ model comparison cho metric: {metric}")
    
    def plot_all_metrics_comparison(
        self,
        results_df: pd.DataFrame
    ) -> None:
        """
        Vẽ so sánh tất cả metrics của các models trong một figure.
        
        Args:
            results_df (pd.DataFrame): DataFrame chứa kết quả của các models
        """
        metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            ascending = metric != 'R2'
            results_sorted = results_df.sort_values(metric, ascending=ascending)
            
            colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_sorted))]
            
            bars = ax.barh(
                results_sorted.index,
                results_sorted[metric],
                color=colors,
                edgecolor='black',
                linewidth=1.5
            )
            
            for bar in bars:
                width = bar.get_width()
                label_format = '{:,.2f}' if metric != 'R2' else '{:.4f}'
                ax.text(
                    width,
                    bar.get_y() + bar.get_height() / 2,
                    ' ' + label_format.format(width),
                    ha='left',
                    va='center',
                    fontsize=9,
                    fontweight='bold'
                )
            
            ax.set_xlabel(metric, fontsize=11)
            ax.set_ylabel('Model', fontsize=11)
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(
            'Comprehensive Model Performance Comparison',
            fontsize=16,
            fontweight='bold',
            y=1.00
        )
        plt.tight_layout()
        
        logger.info("Đã vẽ comprehensive model comparison")
    
    def plot_learning_curve(
        self,
        train_scores: List[float],
        val_scores: List[float],
        model_name: str
    ) -> None:
        """
        Vẽ learning curve (training vs validation performance).
        
        Args:
            train_scores (List[float]): Training scores theo epochs
            val_scores (List[float]): Validation scores theo epochs
            model_name (str): Tên model
        """
        plt.figure(figsize=self.figsize)
        
        epochs = range(1, len(train_scores) + 1)
        
        plt.plot(epochs, train_scores, 'b-o', label='Training Score', linewidth=2, markersize=6)
        plt.plot(epochs, val_scores, 'r-s', label='Validation Score', linewidth=2, markersize=6)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(
            f'Learning Curve - {model_name.upper()}',
            fontsize=14,
            fontweight='bold'
        )
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        logger.info(f"Đã vẽ learning curve cho {model_name}")
    
    def save_plot(
        self,
        filepath: str,
        dpi: Optional[int] = None
    ) -> None:
        """
        Lưu plot hiện tại vào file.
        
        Args:
            filepath (str): Đường dẫn file để lưu
            dpi (int, optional): DPI của ảnh. Mặc định từ config.
        """
        if dpi is None:
            dpi = self.dpi
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(
            filepath,
            dpi=dpi,
            bbox_inches='tight',
            format=self.viz_config['save_format']
        )
        
        logger.info(f"Đã lưu plot vào: {filepath}")
    
    def close_all(self) -> None:
        """
        Đóng tất cả các figures để giải phóng memory.
        """
        plt.close('all')
        logger.info("Đã đóng tất cả figures")
    
    def __repr__(self) -> str:
        """String representation của Visualizer."""
        return (
            f"Visualizer("
            f"save_dir={self.save_dir}, "
            f"figsize={self.figsize}, "
            f"dpi={self.dpi}"
            f")"
        )

if __name__ == "__main__":
    viz = Visualizer()
    
    y_true = np.random.normal(100000, 50000, 1000)
    y_pred = y_true + np.random.normal(0, 20000, 1000)
    
    viz.plot_target_distribution(y_true, "Revenue Distribution")
    viz.save_plot("visualizations/eda_plots/revenue_dist.png")
    viz.close_all()
    
    viz.plot_actual_vs_predicted(y_true, y_pred, "RandomForest")
    viz.save_plot("visualizations/model_results/actual_vs_pred_rf.png")
    viz.close_all()
    
    viz.plot_residuals(y_true, y_pred, "RandomForest")
    viz.save_plot("visualizations/model_results/residuals_rf.png")
    viz.close_all()