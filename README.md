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
## Yêu cầu hệ thống

- **Python**: 3.11  
- **Anaconda / Miniconda** (khuyến nghị)  
- **Git**  
- Windows 10/11 (đã test)

## Cài Đặt

### 1. Clone Dự Án
```bash
git clone https://github.com/PivePipioipia/python-final-project-ds
cd python-final-project-ds
```

### 2. Thiết Lập Môi Trường (Khuyên dùng Conda hoặc Venv)
(Khuyến nghị dùng Anaconda Prompt để ổn định nhất)

```bash
conda create -n movie python=3.11 -y
conda activate movie
```

### 3. Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

### 4. Nếu dùng Command Prompt thường → cần chạy activate.bat trước:
```bash
...anaconda3\Scripts\activate.bat
conda activate movie
```


### 5. Cấu Hình API Key
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

| Đặc Điểm | V1 (Basic) | V2 (Advanced) | Sự Khác Biệt |
| :--- | :--- | :--- | :--- |
| **Xử lý Outlier** | Loại bỏ (IQR Method) | Giữ lại (RobustScaler) | V1 loại bỏ các giá trị ngoại lai; V2 giữ lại toàn bộ dữ liệu. |
| **Số lượng Features** | 65 | 419 | V2 có số chiều dữ liệu lớn hơn nhiều do sử dụng Embeddings. |
| **R2 Score** | ~0.77 | ~0.73 | Kết quả R2 trên tập kiểm thử (Test set). |
| **MAE** | ~$51.5M | ~$49.8M | Sai số tuyệt đối trung bình trên tập kiểm thử. |

Bảng trên tóm tắt sự khác biệt về phương pháp tiếp cận và kết quả thực nghiệm giữa hai phiên bản pipeline.

---

## Cấu Trúc Dự Án

*   **`configs/config.yaml`**: "Bộ não" của dự án. Chứa tham số cấu hình toàn cục, hyperparams và đường dẫn.
*   **`src/`**: Mã nguồn chính.
    *   `data_loader.py`: Thu thập dữ liệu từ API và quản lý file raw.
    *   `preprocessing.py`: Pipeline V1 (Xử lý cơ bản, xóa outlier).
    *   `preprocessing_v2.py`: **(Core)** Pipeline V2 (Nâng cao, giữ outlier, Embeddings).
    *   `model_trainer.py`: Quản lý huấn luyện, Cross-Validation và AutoML (Optuna).
    *   `visualizer.py`: Module chuyên biệt cho vẽ biểu đồ EDA và đánh giá Model.
*   **`notebooks/`**:
    *   `preview_pipeline.ipynb`: Demo chạy pipeline và so sánh hiệu năng V1/V2.
    *   `eda_analysis_final.ipynb`: Phân tích khám phá dữ liệu chuyên sâu (Detailed EDA).
    *   `demo_inference.ipynb`: Demo suy luận (Inference) cho phim mới.
*   **Root**:
    *   `main.py`: CLI Entrypoint - Chạy pipeline, train model, visualize từ dòng lệnh.
    *   `app.py`: Streamlit Web App - Giao diện demo trực quan cho người dùng.
    *   `run_app.bat`: Script tiện ích để khởi chạy nhanh Web App.

---


© 2025 Movie Revenue Prediction Project.
