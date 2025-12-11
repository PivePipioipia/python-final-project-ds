# 🎬 Movie Revenue Prediction (Advanced)

Dự án Machine Learning dự đoán doanh thu phim điện ảnh (Box Office Revenue) 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## Tính Năng Nổi Bật (Highlights)

*   **Dữ Liệu Thực Tế**: Tự động lấy dữ liệu từ **TMDb API** (giai đoạn 2010-2024).
*   **Dual Data Pipeline**: Hỗ trợ 2 chiến lược xử lý dữ liệu song song để so sánh:
    *   **V1 (Basic)**: Fill thiếu bằng trung bình, lọc bỏ outliers (bom tấn), dùng TF-IDF đơn giản.
    *   **V2 (Advanced - Recommended)**:
        *   **KNN Imputer**: Điền dữ liệu thiếu thông minh dựa trên các phim tương đồng.
        *   **Semantic Embeddings (BGE)**: Hiểu nội dung tóm tắt phim (Overview) bằng mô hình ngôn ngữ `BAAI/bge-small-en-v1.5` thay vì đếm từ (Bag-of-Words).
        *   **Robust Scaler**: Xử lý tốt các phim "bom tấn" (Outliers) mà không cần xóa bỏ chúng, giữ lại dữ liệu quý giá.
*   **Tự Động Tối Ưu (AutoML)**: Sử dụng **Optuna** để dò tìm bộ tham số tốt nhất cho RandomForest, XGBoost, LightGBM.
*   **End-to-End Pipeline**: Từ `Raw Data` -> `Feature Engineering` -> `Training` -> `Evaluation`

---

## Cài Đặt

### 1. Clone Dự Án
```bash
git clone https://github.com/PivePipioipia/python-final-project-ds
cd python-final-project-ds
```

### 2. Thiết Lập Môi Trường (Khuyên dùng Conda hoặc Venv)
```bash
# Tạo môi trường
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Mac/Linux)
source venv/bin/activate
```

### 3. Cài Đặt Thư Viện
Dự án yêu cầu các thư viện ML cơ bản và `sentence-transformers` cho NLP.
```bash
pip install -r requirements.txt
```

### 4. Cấu Hình API Key
Tạo file `.env` trong thư mục gốc và điền key của bạn vào:
```env
TMDB_API_KEY=your_api_key_here
```
*(Nếu không có API Key, bạn có thể dùng file dữ liệu mẫu có sẵn trong `data/raw`)*

---

## Hướng Dẫn Chạy (Quick Start)

Cách nhanh nhất để trải nghiệm dự án là chạy Notebook Pipeline.

1.  Mở Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Mở file **`notebooks/preview_pipeline.ipynb`**.
3.  Bấm **Run All**.
    *   Notebook sẽ tự động tải dữ liệu, chạy cả V1 và V2, sau đó in ra bảng so sánh hiệu năng trực tiếp.

---

## Chạy Pipeline với Main Script

Ngoài Jupyter Notebook, bạn có thể chạy toàn bộ pipeline hoặc từng bước riêng lẻ bằng script `main.py`:

### 1. Chạy Toàn Bộ Pipeline (Khuyến Nghị)
```bash
python main.py full-pipeline
```
Lệnh này sẽ tự động:
- Tải dữ liệu từ TMDb API (nếu chưa có)
- Tiền xử lý dữ liệu
- Huấn luyện tất cả các models (Random Forest, XGBoost, LightGBM)
- Tạo visualizations

### 2. Chạy Từng Bước Riêng Lẻ

#### Bước 1: Tải Dữ Liệu
```bash
python main.py fetch-data --start-year 2010 --end-year 2024
```

#### Bước 2: Tiền Xử Lý Dữ Liệu
```bash
python main.py preprocess --input data/raw/movies_2010_2024.csv
```

#### Bước 3: Huấn Luyện Model
```bash
# Train tất cả models
python main.py train --model all

# Hoặc train một model cụ thể
python main.py train --model xgboost
python main.py train --model random_forest
python main.py train --model lightgbm
```

#### Bước 4: Đánh Giá Model
```bash
python main.py evaluate --model-path models/xgboost.pkl
```

#### Bước 5: Tạo Visualizations
```bash
# Tạo tất cả các biểu đồ
python main.py visualize --plot-type all

# Hoặc chỉ tạo EDA plots
python main.py visualize --plot-type eda

# Hoặc chỉ tạo model result plots
python main.py visualize --plot-type model
```

### 3. Kết Quả
Sau khi chạy, kết quả sẽ được lưu tại:
- **Models**: `models/` - Các model đã train (.pkl files)
- **Results**: `results/` - Metrics, predictions, model comparison
- **Logs**: `results/logs/` - Training logs và main logs
- **Visualizations**: `visualizations/` - Các biểu đồ phân tích

---

## Kết Quả So Sánh (Benchmark)

Tại sao lại cần phiên bản V2? Dưới đây là kết quả thực nghiệm trên tập dữ liệu phim 2010-2024:

| Metric | V1 (Basic) | V2 (Advanced) | Nhận Xét |
| :--- | :--- | :--- | :--- |
| **Chiến lược Outlier** | Xóa bỏ phim > 1.5 IQR | Giữ lại (Dùng RobustScaler) | V1 mất hết các phim bom tấn (Marvel, Avatar...), V2 giữ lại được. |
| **Feature Text** | TF-IDF (100 features) | BGE Embeddings (384 dims) | V2 hiểu ngữ nghĩa tốt hơn nhiều. |
| **R2 Score** | ~0.59 | **~0.76** | **V2 giải thích được 76% sự biến thiên dữ liệu.** |
| **MAE (Sai số)** | Thấp ($28M) | Cao ($53M) | V1 sai số thấp do chỉ đoán phim nhỏ. V2 sai số cao hơn do phải đoán cả phim tỷ đô (sai số tuyệt đối lớn là bình thường). |

 **Kết luận**: V2 vượt trội hoàn toàn về khả năng tổng quát hóa và độ chính xác thực tế.

---

## Cấu Trúc Dự Án

*   **`configs/config.yaml`**: "Bộ não" của dự án. Chỉnh sửa năm lấy dữ liệu, tham số model, ngưỡng lọc outlier tại đây.
*   **`src/`**: Mã nguồn chính.
    *   `data_loader.py`: Class `TMDbDataLoader` tải và lưu trữ dữ liệu.
    *   `preprocessing_v2.py`: **(Core)** Class `DataPreprocessorV2` chứa toàn bộ logic xử lý nâng cao.
    *   `model_trainer.py`: Class `ModelTrainer` quản lý việc huấn luyện và Optuna.
*   **`notebooks/`**:
    *   `preview_pipeline.ipynb`: Demo chạy toàn bộ quy trình.
    *   `eda_analysis.ipynb`: Phân tích khám phá dữ liệu (Biểu đồ, Insight).
    *   `demo_inference.ipynb`: Nhập thông tin phim bất kỳ -> Dự đoán doanh thu.

---

Kích hoạt môi trường 
C:\Users\PC\anaconda3\Scripts\activate.bat
conda activate movie_v2

© 2025 Movie Revenue Prediction Project.
