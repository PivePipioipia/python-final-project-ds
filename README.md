# 🎬 Movie Revenue Prediction

Dự án Machine Learning dự đoán doanh thu phim điện ảnh dựa trên các đặc trưng như kinh phí, thể loại, diễn viên, và tóm tắt nội dung (overview).

## Tính Năng Chính
- **Tự động thu thập dữ liệu**: Tích hợp TMDb API để tải dữ liệu phim mới nhất.
- **Data Pipeline**: Quy trình khép kín từ Raw Data -> Preprocessing -> Feature Engineering.
- **Xử lý ngôn ngữ tự nhiên (NLP)**: Sử dụng TF-IDF để trích xuất đặc trưng từ nội dung phim (Overview).
- **Tối ưu hóa Hyperparameter**: Tự động tinh chỉnh tham số cho các mô hình (RandomForest, XGBoost, LightGBM) sử dụng **Optuna**.
- **Giao diện dòng lệnh (CLI)**: Dễ dàng chạy và quản lý pipeline thông qua `main.py`.

## Cài Đặt

1. **Clone dự án**
```bash
git clone https://github.com/PivePipioipia/python-final-project-ds
```

2. **Tạo môi trường ảo**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Cài đặt thư viện**
```bash
pip install -r requirements.txt
```

4. **Cấu hình API Key**

Lấy api key từ web
- TMDB_API_KEY=your_api_key_here


## Hướng Dẫn Sử Dụng

Dự án được điều khiển thông qua file `main.py`. Các lệnh hỗ trợ:

### 1. Thu thập dữ liệu
Tải dữ liệu phim theo cấu hình trong `configs/config.yaml`:
```bash
python main.py fetch-data
```

### 2. Tiền xử lý dữ liệu
Làm sạch, tạo features và chuẩn hóa dữ liệu:
```bash
python main.py preprocess
```

### 3. Huấn luyện mô hình
Train từng model cụ thể hoặc tất cả:
```bash
# Train Random Forest
python main.py train --model random_forest

# Train tất cả và so sánh
python main.py train-all
```

### 4. Chạy toàn bộ Pipeline
Chạy từ A-Z (Fetch -> Preprocess -> Train -> Evaluate):
```bash
python main.py full-pipeline
```

## Cấu Trúc Dự Án

```
movie-revenue-prediction/
├── configs/             # File cấu hình (YAML)
├── data/                # Dữ liệu
│   ├── raw/             # Dữ liệu thô từ API
│   └── processed/       # Dữ liệu đã làm sạch
├── models/              # Các model đã train (.pkl)
├── notebooks/           # Jupyter notebooks cho EDA & Demo
├── results/             # Logs và kết quả thí nghiệm
├── src/                 # Source code chính
│   ├── data_loader.py   # Code tải dữ liệu
│   ├── preprocessing.py # Code xử lý dữ liệu
│   ├── model_trainer.py # Code huấn luyện model
│   └── visualizer.py    # Code vẽ biểu đồ
├── main.py              # Entry point của dự án
├── requirements.txt     # Danh sách thư viện
└── README.md            # Tài liệu dự án
```

## Mô Hình & Hiệu Năng
Hiện tại dự án hỗ trợ 3 thuật toán chính:
- **Random Forest**: Mạnh mẽ, ít bị overfit.
- **XGBoost**: Tốc độ cao, hiệu năng tốt trên dữ liệu bảng.
- **LightGBM**: Tối ưu cho dữ liệu lớn.

Tất cả mô hình đều được đánh giá bằng **RMSE**, **MAE**, **R2** và **MAPE**.
