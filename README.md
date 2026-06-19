Hệ Thống Phân Loại Mức Độ Chín Của Trái Cây 🍎

Dự án ứng dụng Xử lý ảnh và học máy truyền thống để nhận diện mức độ chín của trái cây.

🚀 Công nghệ sử dụng :

Ngôn ngữ: Python

Xử lý ảnh: OpenCV (Phân ngưỡng Otsu, Morphology, Trích xuất đặc trưng HSV, LBP, Shape)

Học máy: Scikit-learn (Support Vector Machine - SVM)

Giao diện Web: Streamlit

⚙️ Cài đặt và Chạy thử

1. Cài đặt thư viện:
pip install -r requirements.txt

2. Chuẩn bị dữ liệu:
Tạo thư mục Train/ ở thư mục gốc, bên trong tạo 3 thư mục con: Unripe, Ripe, Overipe .

3. Chạy ứng dụng Web:
streamlit run app.py
