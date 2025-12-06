"""
Main Entry Point cho Movie Revenue Prediction Project

Script này orchestrate toàn bộ pipeline từ data fetching, preprocessing,
model training, đến visualization. Sử dụng argparse để nhận commands từ CLI.

Usage:
    python main.py fetch-data
    python main.py preprocess
    python main.py train --model random_forest
    python main.py train-all
    python main.py evaluate
    python main.py visualize
    python main.py full-pipeline
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Import các module từ src
from src.data_loader import TMDbDataLoader
from src.preprocessing import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.visualizer import Visualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def fetch_data(args):
    """
    Fetch dữ liệu từ TMDb API và lưu vào file CSV.
    
    Args:
        args: Arguments từ argparse
    """
    logger.info("="*60)
    logger.info("BẮT ĐẦU FETCH DATA TỪ TMDB API")
    logger.info("="*60)
    
    try:
        # Khởi tạo loader
        loader = TMDbDataLoader()
        
        # Fetch movies
        loader.fetch_movies(
            start_year=args.start_year,
            end_year=args.end_year
        )
        
        # Lưu vào CSV
        output_path = args.output or "data/raw/movies_2020_2024.csv"
        loader.save_to_csv(output_path)
        
        logger.info("Đã hoàn thành fetch data thành công")
        
    except Exception as e:
        logger.error(f"Lỗi khi fetch data: {e}")
        sys.exit(1)


def preprocess_data(args):
    """
    Preprocess dữ liệu thô và lưu vào processed data.
    
    Args:
        args: Arguments từ argparse
    """
    logger.info("="*60)
    logger.info("BẮT ĐẦU PREPROCESSING DATA")
    logger.info("="*60)
    
    try:
        # Load raw data
        input_path = args.input or "data/raw/movies_2020_2024.csv"
        logger.info(f"Loading data từ: {input_path}")
        df = pd.read_csv(input_path)
        
        # Khởi tạo preprocessor
        preprocessor = DataPreprocessor()
        
        # Split data trước khi preprocess
        # (để tránh data leakage khi remove outliers)
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
        
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        # Fit và transform training data
        X_train, y_train = preprocessor.fit_transform(train_df)
        
        # Transform test data (không remove outliers)
        X_test, y_test = preprocessor.transform(test_df, remove_outliers=False)
        
        # Lưu processed data
        output_dir = args.output or "data/processed"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        preprocessor.save_processed_data(
            X_train, 
            f"{output_dir}/X_train.csv",
            y_train
        )
        preprocessor.save_processed_data(
            X_test,
            f"{output_dir}/X_test.csv",
            y_test
        )
        
        # Lưu preprocessor object
        preprocessor.save_preprocessor("models/preprocessor.pkl")
        
        logger.info("Đã hoàn thành preprocessing thành công")
        logger.info(f"  X_train shape: {X_train.shape}")
        logger.info(f"  X_test shape: {X_test.shape}")
        logger.info(f"  Number of features: {len(preprocessor.get_feature_names())}")
        
    except Exception as e:
        logger.error(f"Lỗi khi preprocessing: {e}")
        sys.exit(1)


def train_model(args):
    """
    Train một hoặc tất cả các models.
    
    Args:
        args: Arguments từ argparse
    """
    logger.info("="*60)
    logger.info("BẮT ĐẦU TRAINING MODELS")
    logger.info("="*60)
    
    try:
        # Load processed data
        data_dir = args.data_dir or "data/processed"
        logger.info(f"Loading processed data từ: {data_dir}")
        
        # Load train và test data từ CSV
        train_data = pd.read_csv(f"{data_dir}/X_train.csv")
        test_data = pd.read_csv(f"{data_dir}/X_test.csv")
        
        # Tách X và y
        X_train = train_data.drop(columns=['revenue']).values
        y_train = train_data['revenue'].values
        X_test = test_data.drop(columns=['revenue']).values
        y_test = test_data['revenue'].values
        
        # Khởi tạo trainer
        trainer = ModelTrainer()
        trainer.load_data(X_train, X_test, y_train, y_test)
        
        # Train models
        if args.model == 'all':
            trainer.train_all_models()
        else:
            trainer.train_model(args.model)
        
        # So sánh models (nếu train nhiều models)
        if len(trainer.models) > 1:
            comparison_df = trainer.compare_models()
            comparison_df.to_csv("results/model_comparison.csv")
        
        # Lưu models và results
        trainer.save_all_models(args.output or "models")
        trainer.save_results()
        trainer.save_best_params()
        
        logger.info("Đã hoàn thành training thành công")
        
    except Exception as e:
        logger.error(f"Lỗi khi training: {e}")
        sys.exit(1)


def evaluate_model(args):
    """
    Evaluate model đã train trên test set.
    
    Args:
        args: Arguments từ argparse
    """
    logger.info("="*60)
    logger.info("BẮT ĐẦU EVALUATION")
    logger.info("="*60)
    
    try:
        # Load model
        model_path = args.model_path or "models/random_forest.pkl"
        logger.info(f"Loading model từ: {model_path}")
        model = ModelTrainer.load_model(model_path)
        
        # Load test data
        data_dir = args.data_dir or "data/processed"
        test_data = pd.read_csv(f"{data_dir}/X_test.csv")
        
        X_test = test_data.drop(columns=['revenue']).values
        y_test = test_data['revenue'].values
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        
        logger.info("\nTest Set Evaluation Results:")
        logger.info(f"  RMSE: ${rmse:,.2f}")
        logger.info(f"  MAE:  ${mae:,.2f}")
        logger.info(f"  R2:   {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        # Lưu predictions
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'error': y_test - y_pred
        })
        
        output_path = args.output or "results/predictions.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nĐã lưu predictions vào: {output_path}")
        
    except Exception as e:
        logger.error(f"Lỗi khi evaluation: {e}")
        sys.exit(1)


def visualize_results(args):
    """
    Tạo tất cả visualizations cho EDA và model results.
    
    Args:
        args: Arguments từ argparse
    """
    logger.info("="*60)
    logger.info("BẮT ĐẦU VISUALIZATION")
    logger.info("="*60)
    
    try:
        # Khởi tạo visualizer
        viz = Visualizer()
        
        # Load data
        data_dir = args.data_dir or "data/processed"
        
        # 1. EDA Plots
        if args.plot_type in ['all', 'eda']:
            logger.info("Tạo EDA plots...")
            
            train_data = pd.read_csv(f"{data_dir}/X_train.csv")
            y_train = train_data['revenue'].values
            
            # Target distribution
            viz.plot_target_distribution(y_train, "Revenue Distribution")
            viz.save_plot("visualizations/eda_plots/revenue_distribution.png")
            viz.close_all()
            
            # Correlation matrix
            viz.plot_correlation_matrix(train_data, "Feature Correlations")
            viz.save_plot("visualizations/eda_plots/correlation_matrix.png")
            viz.close_all()
            
            logger.info("Đã tạo EDA plots")
        
        # 2. Model Results Plots
        if args.plot_type in ['all', 'model']:
            logger.info("Tạo model result plots...")
            
            # Load test data và predictions
            test_data = pd.read_csv(f"{data_dir}/X_test.csv")
            X_test = test_data.drop(columns=['revenue']).values
            y_test = test_data['revenue'].values
            
            # Load models và generate plots
            models = ['random_forest', 'xgboost', 'lightgbm']
            model_dir = args.model_dir or "models"
            
            for model_name in models:
                model_path = f"{model_dir}/{model_name}.pkl"
                
                if not Path(model_path).exists():
                    logger.warning(f"Không tìm thấy model: {model_path}")
                    continue
                
                logger.info(f"Generating plots cho {model_name}...")
                
                # Load model
                model = ModelTrainer.load_model(model_path)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Actual vs Predicted
                viz.plot_actual_vs_predicted(y_test, y_pred, model_name)
                viz.save_plot(f"visualizations/model_results/actual_vs_pred_{model_name}.png")
                viz.close_all()
                
                # Residual Analysis
                viz.plot_residuals(y_test, y_pred, model_name)
                viz.save_plot(f"visualizations/model_results/residuals_{model_name}.png")
                viz.close_all()
                
                # Feature Importance
                if hasattr(model, 'feature_importances_'):
                    preprocessor = DataPreprocessor.load_preprocessor("models/preprocessor.pkl")
                    feature_names = preprocessor.get_feature_names()
                    
                    trainer = ModelTrainer()
                    trainer.models[model_name] = model
                    
                    importance_df = trainer.get_feature_importance(
                        model_name,
                        feature_names
                    )
                    
                    viz.plot_feature_importance(importance_df, model_name)
                    viz.save_plot(f"visualizations/model_results/feature_importance_{model_name}.png")
                    viz.close_all()
            
            # Model Comparison
            results_path = "results/model_comparison.csv"
            if Path(results_path).exists():
                results_df = pd.read_csv(results_path, index_col=0)
                
                viz.plot_all_metrics_comparison(results_df)
                viz.save_plot("visualizations/model_results/model_comparison_all_metrics.png")
                viz.close_all()
                
                logger.info("Đã tạo model comparison plots")
        
        logger.info("Đã hoàn thành tất cả visualizations")
        
    except Exception as e:
        logger.error(f"Lỗi khi visualization: {e}")
        sys.exit(1)


def run_full_pipeline(args):
    """
    Chạy toàn bộ pipeline từ đầu đến cuối.
    
    Args:
        args: Arguments từ argparse
    """
    logger.info("="*60)
    logger.info("BẮT ĐẦU FULL PIPELINE")
    logger.info("="*60)
    
    try:
        # 1. Fetch Data (nếu chưa có)
        raw_data_path = "data/raw/movies_2020_2024.csv"
        if not Path(raw_data_path).exists() or args.force_fetch:
            logger.info("\nStep 1: Fetching data from TMDb API...")
            args.start_year = 2020
            args.end_year = 2024
            args.output = raw_data_path
            fetch_data(args)
        else:
            logger.info(f"\nStep 1: Skipping fetch (data exists at {raw_data_path})")
        
        # 2. Preprocess
        logger.info("\nStep 2: Preprocessing data...")
        args.input = raw_data_path
        args.output = "data/processed"
        preprocess_data(args)
        
        # 3. Train All Models
        logger.info("\nStep 3: Training all models...")
        args.model = 'all'
        args.data_dir = "data/processed"
        args.output = "models"
        train_model(args)
        
        # 4. Visualize
        logger.info("\nStep 4: Creating visualizations...")
        args.plot_type = 'all'
        args.data_dir = "data/processed"
        args.model_dir = "models"
        visualize_results(args)
        
        logger.info("\n" + "="*60)
        logger.info("ĐÃ HOÀN THÀNH FULL PIPELINE THÀNH CÔNG")
        logger.info("="*60)
        logger.info("\nKết quả được lưu tại:")
        logger.info("  - Models: models/")
        logger.info("  - Results: results/")
        logger.info("  - Visualizations: visualizations/")
        
    except Exception as e:
        logger.error(f"Lỗi trong full pipeline: {e}")
        sys.exit(1)


def main():
    """
    Main function để parse arguments và chạy commands tương ứng.
    """
    parser = argparse.ArgumentParser(
        description="Movie Revenue Prediction - Data Science Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch dữ liệu từ TMDb API
  python main.py fetch-data --start-year 2020 --end-year 2024
  
  # Preprocess dữ liệu
  python main.py preprocess --input data/raw/movies_2020_2024.csv
  
  # Train một model cụ thể
  python main.py train --model random_forest
  
  # Train tất cả models
  python main.py train --model all
  
  # Evaluate model
  python main.py evaluate --model-path models/xgboost.pkl
  
  # Tạo visualizations
  python main.py visualize --plot-type all
  
  # Chạy toàn bộ pipeline
  python main.py full-pipeline
        """
    )
    
    # Tạo subparsers cho các commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Fetch Data Command
    fetch_parser = subparsers.add_parser(
        'fetch-data',
        help='Fetch movie data từ TMDb API'
    )
    fetch_parser.add_argument(
        '--start-year',
        type=int,
        default=2020,
        help='Năm bắt đầu (default: 2020)'
    )
    fetch_parser.add_argument(
        '--end-year',
        type=int,
        default=2024,
        help='Năm kết thúc (default: 2024)'
    )
    fetch_parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: data/raw/movies_2020_2024.csv)'
    )
    
    # Preprocess Command
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Preprocess raw data'
    )
    preprocess_parser.add_argument(
        '--input',
        type=str,
        help='Input raw data file (default: data/raw/movies_2020_2024.csv)'
    )
    preprocess_parser.add_argument(
        '--output',
        type=str,
        help='Output directory cho processed data (default: data/processed)'
    )
    
    # Train Command
    train_parser = subparsers.add_parser(
        'train',
        help='Train machine learning models'
    )
    train_parser.add_argument(
        '--model',
        type=str,
        choices=['random_forest', 'xgboost', 'lightgbm', 'all'],
        default='all',
        help='Model to train (default: all)'
    )
    train_parser.add_argument(
        '--data-dir',
        type=str,
        help='Processed data directory (default: data/processed)'
    )
    train_parser.add_argument(
        '--output',
        type=str,
        help='Output directory cho models (default: models)'
    )
    
    # Evaluate Command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate trained model'
    )
    eval_parser.add_argument(
        '--model-path',
        type=str,
        help='Path to model file (default: models/random_forest.pkl)'
    )
    eval_parser.add_argument(
        '--data-dir',
        type=str,
        help='Processed data directory (default: data/processed)'
    )
    eval_parser.add_argument(
        '--output',
        type=str,
        help='Output file cho predictions (default: results/predictions.csv)'
    )
    
    # Visualize Command
    viz_parser = subparsers.add_parser(
        'visualize',
        help='Generate visualizations'
    )
    viz_parser.add_argument(
        '--plot-type',
        type=str,
        choices=['all', 'eda', 'model'],
        default='all',
        help='Type of plots to generate (default: all)'
    )
    viz_parser.add_argument(
        '--data-dir',
        type=str,
        help='Processed data directory (default: data/processed)'
    )
    viz_parser.add_argument(
        '--model-dir',
        type=str,
        help='Models directory (default: models)'
    )
    
    # Full Pipeline Command
    full_parser = subparsers.add_parser(
        'full-pipeline',
        help='Chạy toàn bộ pipeline từ đầu đến cuối'
    )
    full_parser.add_argument(
        '--force-fetch',
        action='store_true',
        help='Force fetch data ngay cả khi đã tồn tại'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Kiểm tra command
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Chạy command tương ứng
    if args.command == 'fetch-data':
        fetch_data(args)
    elif args.command == 'preprocess':
        preprocess_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'visualize':
        visualize_results(args)
    elif args.command == 'full-pipeline':
        run_full_pipeline(args)


if __name__ == "__main__":
    main()