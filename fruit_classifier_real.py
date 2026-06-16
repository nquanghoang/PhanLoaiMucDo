import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


RIPENESS_LABELS = {
    0: "Xanh (Unripe)",
    1: "Chín (Ripe)",
    2: "Chín quá (Overripe)"
}

def load_real_dataset(dataset_path: str) -> tuple:
    """
    Đọc ảnh thật từ thư mục được phân loại sẵn (3 lớp).
    Cấu trúc yêu cầu: dataset_path / tên_thư_mục_nhãn / ảnh.jpg
    """
    print(f"[INFO] Đang quét thư mục dữ liệu thật: {dataset_path} ...")
    images = []
    labels = []
    
    
    folder_to_label = {
        "Unripe": 0,
        "Ripe": 1,
        "Overipe": 2
    }

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {dataset_path}. Vui lòng tạo thư mục và cho ảnh vào.")

    for folder_name, label_id in folder_to_label.items():
        folder_path = os.path.join(dataset_path, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"  [CẢNH BÁO] Không thấy thư mục {folder_name}, bỏ qua...")
            continue
            
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    images.append(img)
                    labels.append(label_id)
                    count += 1
                else:
                    print(f"  [LỖI] Không thể đọc file ảnh: {filename}")
                    
        print(f"  -> Đã tải {count} ảnh từ thư mục [{folder_name}]")

    print(f"[INFO] Tải hoàn tất! Tổng số ảnh thật: {len(images)}")
    return images, np.array(labels)


def preprocess_image(img: np.ndarray, target_size: tuple = (128, 128)) -> np.ndarray:
    """ Tiền xử lý ảnh: resize, lọc nhiễu, chuẩn hóa. """
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def segment_fruit(img: np.ndarray) -> np.ndarray:
    """ Phân đoạn vùng trái cây bằng phân ngưỡng Otsu + morphology. """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask


def extract_color_features_hsv(img: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    if mask is not None and mask.sum() > 0:
        pixels_h = hsv[:, :, 0][mask > 0]
        pixels_s = hsv[:, :, 1][mask > 0]
        pixels_v = hsv[:, :, 2][mask > 0]
    else:
        pixels_h = hsv[:, :, 0].flatten()
        pixels_s = hsv[:, :, 1].flatten()
        pixels_v = hsv[:, :, 2].flatten()

    features = []
    for ch_pixels, n_bins, ch_range in [(pixels_h, 18, (0, 180)), (pixels_s, 8, (0, 256)), (pixels_v, 8, (0, 256))]:
        features.extend([np.mean(ch_pixels), np.std(ch_pixels)])
        hist, _ = np.histogram(ch_pixels, bins=n_bins, range=ch_range)
        hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
        features.extend(hist.tolist())
    return np.array(features)

def extract_color_features_lab(img: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    features = []
    for c in range(3):
        ch = lab[:, :, c]
        pixels = ch[mask > 0] if (mask is not None and mask.sum() > 0) else ch.flatten()
        features.extend([np.mean(pixels), np.std(pixels), np.percentile(pixels, 25), np.percentile(pixels, 75)])
    return np.array(features)

def extract_texture_features_lbp(img: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=16, R=2, method='uniform')
    lbp_pixels = lbp[mask > 0] if (mask is not None and mask.sum() > 0) else lbp.flatten()
    hist, _ = np.histogram(lbp_pixels, bins=18, range=(0, 18))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
    return hist

def extract_shape_features(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(5)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / (h + 1e-6)
    extent = area / (w * h + 1e-6)
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)
    img_area = mask.shape[0] * mask.shape[1]
    area_ratio = area / (img_area + 1e-6)
    return np.array([area_ratio, aspect_ratio, extent, circularity, solidity])

def extract_all_features(img: np.ndarray) -> np.ndarray:
    img = preprocess_image(img)
    mask = segment_fruit(img)
    f_hsv = extract_color_features_hsv(img, mask)
    f_lab = extract_color_features_lab(img, mask)
    f_lbp = extract_texture_features_lbp(img, mask)
    f_shape = extract_shape_features(mask)
    return np.concatenate([f_hsv, f_lab, f_lbp, f_shape])

def build_models() -> dict:
    models = {
        "SVM (RBF Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True))
        ]),
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'))
        ])
    }
    return models

def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n[INFO] Tập huấn luyện: {len(X_train)} mẫu | Tập kiểm tra: {len(X_test)} mẫu")

    models = build_models()
    results = {}

    for name, model in models.items():
        print(f"\n  [{name}] Đang huấn luyện...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        except:
            cv_mean, cv_std = 0.0, 0.0

        print(f"    Accuracy:     {acc:.4f}")
        print(f"    F1-score:     {f1:.4f}")
        
        target_names = [RIPENESS_LABELS[i] for i in sorted(list(set(y_test)))]
        
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_test": y_test,
            "accuracy": acc,
            "f1": f1,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "report": classification_report(y_test, y_pred, target_names=target_names)
        }
    return results


def plot_confusion_matrix(y_test, y_pred, model_name, save_path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    classes = [RIPENESS_LABELS[i] for i in range(3)]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=15, ha='right', fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')

    ax.set_ylabel('Nhãn thực tế (Thật)', fontsize=11)
    ax.set_xlabel('Nhãn dự đoán (Máy đoán)', fontsize=11)
    ax.set_title(f'Ma trận nhầm lẫn – {model_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [Saved] {save_path}")

def plot_model_comparison(results, save_path):
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    f1s  = [results[n]["f1"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, accs, width, label='Test Accuracy', color='#2196F3')
    b2 = ax.bar(x + width/2, f1s,  width, label='F1-score',      color='#4CAF50')

    ax.set_ylabel('Điểm', fontsize=11)
    ax.set_title('So sánh hiệu suất các mô hình', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0.0, 1.1)
    ax.legend(fontsize=10)

    for bars in [b1, b2]:
        for bar in bars:
            ax.annotate(f'{bar.get_height():.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [Saved] {save_path}")


def main():
    print("=" * 60)
    print("  HỆ THỐNG PHÂN LOẠI TRÁI CÂY (3 LỚP - DỮ LIỆU THẬT)")
    print("=" * 60)

    DATASET_DIR = "Train" 
    OUTPUT_DIR = "output_figs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        images, labels = load_real_dataset(DATASET_DIR)
    except Exception as e:
        print(f"[LỖI] {e}")
        return

    if len(images) < 10:
        print("\n[CẢNH BÁO] Số lượng ảnh quá ít (<10 ảnh). Vui lòng thêm nhiều ảnh hơn vào các thư mục để train hiệu quả.")
        return

    print("\n[INFO] Đang trích xuất đặc trưng (75 chiều/mẫu) từ ảnh thật...")
    X = np.array([extract_all_features(img) for img in images])
    y = labels
    print(f"[INFO] Ma trận đặc trưng đã tạo: {X.shape}")

    print("\n[INFO] Bắt đầu huấn luyện mô hình...")
    results = train_and_evaluate(X, y)

    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    best_result = results[best_model_name]

    print("\n[INFO] Đang vẽ và lưu biểu đồ đánh giá...")
    plot_confusion_matrix(best_result["y_test"], best_result["y_pred"], best_model_name,
                          os.path.join(OUTPUT_DIR, "fig1_confusion_matrix.png"))
    plot_model_comparison(results, os.path.join(OUTPUT_DIR, "fig2_model_comparison.png"))

    print(f"\n{'='*60}")
    print(f"  MÔ HÌNH TỐT NHẤT: {best_model_name}")
    print(f"  Accuracy trên ảnh thật: {best_result['accuracy']:.4f}")
    print(f"{'='*60}")
    print(f"\n[INFO] Đã lưu các biểu đồ tại thư mục: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()