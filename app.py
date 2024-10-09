import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="Ứng dụng Nâng Cao Chất Lượng Ảnh",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tiêu đề của ứng dụng
st.title("🖼️ Ứng dụng Nâng Cao Chất Lượng Ảnh")

# Giới thiệu
st.markdown("""
Chào mừng bạn đến với ứng dụng nâng cao chất lượng ảnh. Bạn có thể:
- Tải lên một ảnh từ máy tính của bạn.
- Áp dụng các bộ lọc và tính năng để cải thiện và phân tích ảnh.
""")

# Sidebar - Tải ảnh và chọn bộ lọc
st.sidebar.title("🎛️ Tùy chọn")
uploaded_file = st.sidebar.file_uploader("Chọn một ảnh...", type=["jpg", "jpeg", "png"])

# Định nghĩa kích thước hiển thị ảnh cố định
DISPLAY_WIDTH = 300  # Bạn có thể thay đổi giá trị này theo ý muốn

if uploaded_file is not None:
    # Đọc ảnh từ file tải lên
    image = Image.open(uploaded_file)
    # Chuyển đổi ảnh về RGB nếu ảnh là RGBA
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Hiển thị ảnh gốc với kích thước cố định
    st.subheader("Ảnh Gốc")
    resized_image = image.resize((DISPLAY_WIDTH, int(image.height * DISPLAY_WIDTH / image.width)))
    st.image(resized_image, caption='Ảnh Gốc', use_column_width=False)

    # Chọn các bộ lọc
    st.sidebar.subheader("Chọn các bộ lọc muốn áp dụng")

    filter_options = ["Brightness/Contrast", "Histogram Equalization", "CLAHE", "Denoising", "Sharpening", "Blurring", "Edge Detection", "Morphological Operations"]
    selected_filters = st.sidebar.multiselect("Chọn bộ lọc:", filter_options)

    # Khởi tạo biến lưu ảnh hiện tại
    current_image = img.copy()
    processed_images = []
    captions = []

    # Áp dụng các bộ lọc được chọn
    for filter_name in selected_filters:
        if filter_name == "Brightness/Contrast":
            with st.sidebar.expander("Brightness & Contrast"):
                brightness = st.slider("Độ sáng", -100, 100, 0, key='brightness')
                contrast = st.slider("Độ tương phản", -100, 100, 0, key='contrast')

            beta = brightness
            alpha = contrast / 100 + 1  # contrast từ -100 đến 100, alpha từ 0 đến 2

            adjusted = cv2.convertScaleAbs(current_image, alpha=alpha, beta=beta)
            current_image = adjusted
            processed_images.append(cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB))
            captions.append("Brightness/Contrast Adjusted")

        elif filter_name == "Histogram Equalization":
            # Không cần tham số
            # Chuyển đổi sang YCrCb
            img_y_cr_cb = cv2.cvtColor(current_image, cv2.COLOR_BGR2YCrCb)
            # Tách các kênh
            y, cr, cb = cv2.split(img_y_cr_cb)
            # Áp dụng equalization trên kênh Y
            y_eq = cv2.equalizeHist(y)
            # Gộp các kênh lại
            img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
            # Chuyển đổi lại sang BGR
            img_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2BGR)
            current_image = img_eq
            processed_images.append(cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB))
            captions.append("Histogram Equalization")

        elif filter_name == "CLAHE":
            with st.sidebar.expander("CLAHE"):
                clip_limit = st.slider("Clip Limit", 1.0, 10.0, 2.0, key='clip_limit')
                tile_grid_size = st.slider("Tile Grid Size", 1, 16, 8, key='tile_grid_size')

            # Chuyển đổi sang LAB
            lab = cv2.cvtColor(current_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # Áp dụng CLAHE trên kênh L
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            l_clahe = clahe.apply(l)
            # Gộp các kênh lại
            lab_clahe = cv2.merge((l_clahe, a, b))
            # Chuyển đổi lại sang BGR
            img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
            current_image = img_clahe
            processed_images.append(cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB))
            captions.append("CLAHE Applied")

        elif filter_name == "Denoising":
            with st.sidebar.expander("Denoising"):
                h = st.slider('Giá trị h', 0, 50, 10, key='h')
                hColor = st.slider('Giá trị hColor', 0, 50, 10, key='hColor')
                templateWindowSize = st.slider('Kích thước templateWindowSize', 1, 10, 7, key='templateWindowSize')
                searchWindowSize = st.slider('Kích thước searchWindowSize', 1, 30, 21, key='searchWindowSize')

            denoised = cv2.fastNlMeansDenoisingColored(
                current_image, None, h=h, hColor=hColor,
                templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize
            )
            current_image = denoised
            processed_images.append(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
            captions.append("Denoised Image")

        elif filter_name == "Sharpening":
            # Không cần tham số
            kernel_sharpening = np.array([[-1,-1,-1],
                                          [-1, 9,-1],
                                          [-1,-1,-1]])
            sharpened = cv2.filter2D(current_image, -1, kernel_sharpening)
            current_image = sharpened
            processed_images.append(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
            captions.append("Sharpened Image")

        elif filter_name == "Blurring":
            with st.sidebar.expander("Blurring"):
                blur_type = st.selectbox("Chọn loại blur:", ["Average", "Gaussian", "Median", "Bilateral"], key='blur_type')
                if blur_type == "Average":
                    ksize = st.slider('Kích thước kernel', 1, 15, 5, step=2, key='avg_ksize')
                    blurred = cv2.blur(current_image, (ksize, ksize))
                elif blur_type == "Gaussian":
                    ksize = st.slider('Kích thước kernel', 1, 15, 5, step=2, key='gauss_ksize')
                    blurred = cv2.GaussianBlur(current_image, (ksize, ksize), 0)
                elif blur_type == "Median":
                    ksize = st.slider('Kích thước kernel', 1, 15, 5, step=2, key='median_ksize')
                    blurred = cv2.medianBlur(current_image, ksize)
                elif blur_type == "Bilateral":
                    diameter = st.slider('Đường kính pixel', 1, 15, 9, step=2, key='bilat_diameter')
                    sigmaColor = st.slider('Sigma Color', 1, 200, 75, key='bilat_sigmaColor')
                    sigmaSpace = st.slider('Sigma Space', 1, 200, 75, key='bilat_sigmaSpace')
                    blurred = cv2.bilateralFilter(current_image, diameter, sigmaColor, sigmaSpace)
            current_image = blurred
            processed_images.append(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
            captions.append(f"{blur_type} Blurred")

        elif filter_name == "Edge Detection":
            with st.sidebar.expander("Edge Detection"):
                edge_method = st.selectbox("Chọn phương pháp:", ["Sobel", "Canny"], key='edge_method')
                if edge_method == "Sobel":
                    ksize = st.slider('Kích thước kernel', 1, 7, 5, step=2, key='sobel_ksize')
                elif edge_method == "Canny":
                    canny_threshold1 = st.slider('Ngưỡng dưới', 0, 255, 100, key='canny_thresh1')
                    canny_threshold2 = st.slider('Ngưỡng trên', 0, 255, 200, key='canny_thresh2')

            gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

            if edge_method == "Sobel":
                sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
                sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)
                sobel = cv2.magnitude(sobelx, sobely)
                sobel = np.uint8(np.absolute(sobel))
                edge_img = sobel
                captions.append("Sobel Edge")
            elif edge_method == "Canny":
                edge_img = cv2.Canny(gray_image, canny_threshold1, canny_threshold2)
                captions.append("Canny Edge")

            # Chuyển đổi ảnh cạnh sang RGB để hiển thị
            edge_img_rgb = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB)
            current_image = edge_img_rgb
            processed_images.append(edge_img_rgb)
            # Edge detection là bước cuối cùng
            break  # Dừng áp dụng các bộ lọc tiếp theo

        elif filter_name == "Morphological Operations":
            with st.sidebar.expander("Morphological Operations"):
                operation = st.selectbox("Chọn loại:", ["Erosion", "Dilation", "Opening", "Closing"], key='morph_operation')
                kernel_size = st.slider('Kích thước kernel', 1, 15, 5, step=2, key='morph_kernel_size')

            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

            if operation == "Erosion":
                result = cv2.erode(gray_image, kernel, iterations=1)
                captions.append("Erosion")
            elif operation == "Dilation":
                result = cv2.dilate(gray_image, kernel, iterations=1)
                captions.append("Dilation")
            elif operation == "Opening":
                result = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
                captions.append("Opening")
            elif operation == "Closing":
                result = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
                captions.append("Closing")

            # Chuyển đổi ảnh kết quả sang RGB để hiển thị
            result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            current_image = result_rgb
            processed_images.append(result_rgb)

    # Hiển thị ảnh sau khi áp dụng các bộ lọc
    if processed_images:
        st.subheader("Kết quả sau khi áp dụng các bộ lọc")
        num_images = len(processed_images)
        cols = st.columns(num_images)
        for idx, (img_to_show, caption) in enumerate(zip(processed_images, captions)):
            with cols[idx]:
                # Resize ảnh để hiển thị đồng nhất
                display_img = Image.fromarray(img_to_show)
                display_img = display_img.resize((DISPLAY_WIDTH, int(display_img.height * DISPLAY_WIDTH / display_img.width)))
                st.image(display_img, caption=caption, use_column_width=False)
                # Thêm nút tải xuống
                buf = io.BytesIO()
                display_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="📥 Tải xuống",
                    data=byte_im,
                    file_name=f"{caption.replace(' ', '_').lower()}.png",
                    mime="image/png"
                )

else:
    st.write("Vui lòng tải lên một ảnh để bắt đầu.")

# Thêm phần chân trang
st.markdown("""
---
📝 **Hướng dẫn sử dụng**:
1. Tải lên một ảnh từ máy tính của bạn bằng cách sử dụng nút "Browse files" ở thanh bên trái.
2. Chọn các bộ lọc bạn muốn áp dụng.
3. Điều chỉnh các tham số cho từng bộ lọc nếu cần.
4. **Kết quả sẽ tự động cập nhật khi bạn thay đổi tham số.**
5. Bạn có thể tải xuống ảnh đã xử lý bằng cách nhấn nút "Tải xuống" dưới mỗi ảnh.
""")
