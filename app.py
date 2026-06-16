import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

from fruit_classifier_real import (
    RIPENESS_LABELS, 
    load_real_dataset, 
    extract_all_features, 
    build_models,
    preprocess_image,  
    segment_fruit      
)

@st.cache_resource
def get_trained_model():
   
    dataset_dir = "Train"
    if not os.path.exists(dataset_dir):
        st.error(f"Không tìm thấy thư mục '{dataset_dir}'. Vui lòng đảm bảo thư mục Train nằm cùng cấp với file app.py.")
        st.stop()
        
    images, labels = load_real_dataset(dataset_dir)
    if len(images) == 0:
        st.error("Thư mục Train không có ảnh nào. Vui lòng thêm ảnh.")
        st.stop()
        
    X = np.array([extract_all_features(img) for img in images])
    y = np.array(labels)
    
    models = build_models()
    model = models["SVM (RBF Kernel)"]
    model.fit(X, y)
    
    return model


st.set_page_config(page_title="Phân loại Trái Cây", page_icon="🍎", layout="centered")

st.title("🍎 Hệ Thống Phân Loại Mức Độ Chín")
st.markdown("**Đề tài:** Nhận diện mức độ chín của trái cây bằng Xử lý ảnh và Support Vector Machine (SVM).")
st.divider()

with st.spinner("⏳ Đang khởi tạo trí tuệ nhân tạo và học dữ liệu từ thư mục Train..."):
    model = get_trained_model()
st.success("✅ Hệ thống AI đã sẵn sàng!")

st.subheader("1. Kiểm tra trái cây")
uploaded_file = st.file_uploader("Tải một bức ảnh lên đây (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Ảnh bạn vừa tải lên", use_container_width=True)

    with col2:
        st.markdown("### Đang phân tích...")
        with st.spinner("Đang trích xuất 75 đặc trưng (Màu sắc, Kết cấu, Hình dạng)..."):
            img_array = np.array(image.convert('RGB'))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            try:
                features = extract_all_features(img_bgr).reshape(1, -1)
                
                pred_idx = int(model.predict(features)[0])
                probas = model.predict_proba(features)[0]
                
                label = RIPENESS_LABELS[pred_idx]
                confidence = probas[pred_idx] * 100
                
                st.markdown(f"**Kết quả phân loại:**")
                st.success(f"🏷️ **{label}**")
                st.info(f"🎯 **Độ tin cậy:** {confidence:.2f}%")
                
            except Exception as e:
                st.error("Không thể nhận diện được vật thể (trái cây) trong ảnh. Vui lòng thử ảnh có phông nền rõ ràng hơn!")
                st.write(f"Lỗi hệ thống: {e}")
                st.stop()

    st.subheader("2. Chi tiết xác suất dự đoán")
    df_probs = pd.DataFrame({
        "Mức độ": [RIPENESS_LABELS[i] for i in range(3)],
        "Xác suất (%)": probas * 100
    })
    
    st.bar_chart(df_probs.set_index("Mức độ"), color="#4CAF50")

    st.subheader("3. Phân tích quá trình bóc tách ảnh")
    with st.expander("🔍 Bấm vào đây để xem máy tính cắt ảnh như thế nào"):
        st.markdown("Hệ thống sử dụng thuật toán **Otsu** và **Morphology** để tách trái cây ra khỏi nền trước khi trích xuất đặc trưng.")
        
        img_pre = preprocess_image(img_bgr)
        mask = segment_fruit(img_pre)
        
        roi = img_pre.copy()
        roi[mask == 0] = [0, 0, 0]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB), caption="1. Cân bằng sáng", use_container_width=True)
        with c2:
            st.image(mask, caption="2. Mặt nạ Otsu (Mask)", use_container_width=True, clamp=True)
        with c3:
            st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="3. Tách nền (ROI)", use_container_width=True)