# -*- coding: utf-8 -*-
"""
Mã nguồn tổng quan thực hiện lọc ảnh màu trong không gian xyY
bao gồm cả phương pháp theo Paper và phương pháp Naive để so sánh.

Phương pháp theo Paper (xyY_filter_paper):
- Dựa trên công thức (3a), (3b) và mô tả thực nghiệm Mục 2.4 trong báo cáo.
- Áp dụng bộ lọc Gaussian cho các số hạng có trọng số Y/y.
- Giữ nguyên kênh độ sáng Y cuối cùng.

Phương pháp Naive (naive_filter):
- Áp dụng bộ lọc Gaussian trực tiếp cho các kênh sắc độ x và y.
- Giữ nguyên kênh độ sáng Y (để so sánh).
- Thường gây ra dịch chuyển màu như mô tả trong báo cáo.

Code này sử dụng ma trận chuyển đổi sRGB D65 tiêu chuẩn.
Cảnh báo: Hằng số EPSILON bảo vệ chia cho 0 đã bị loại bỏ theo yêu cầu trước đó.
Điều này có thể dẫn đến lỗi hoặc kết quả không chính xác (NaN/Inf) nếu gặp mẫu số bằng 0.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os

# --- Các hàm chuyển đổi không gian màu ---

def rgb_to_xyz(rgb):
    """Chuyển đổi RGB (float 0-1) sang XYZ dùng ma trận sRGB D65."""
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    shape = rgb.shape
    rgb = rgb.astype(np.float64)
    rgb_flat = rgb.reshape(-1, 3)
    xyz_flat = rgb_flat @ M.T
    return xyz_flat.reshape(shape)

def xyz_to_rgb(xyz):
    """Chuyển đổi XYZ sang RGB (float 0-1) dùng ma trận sRGB D65."""
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    shape = xyz.shape
    xyz_flat = xyz.reshape(-1, 3)
    rgb_flat = xyz_flat @ M_inv.T
    # Clip để đảm bảo giá trị nằm trong khoảng [0, 1]
    return np.clip(rgb_flat.reshape(shape), 0, 1)

def xyz_to_xyY(xyz):
    """Chuyển đổi XYZ sang xyY."""
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    denom = X + Y + Z
    # Xử lý chia cho 0 (không dùng EPSILON)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(X, denom)
        y = np.divide(Y, denom)
        x = np.nan_to_num(x) # Thay NaN (do 0/0) bằng 0
        y = np.nan_to_num(y)
    return np.stack((x, y, Y), axis=-1)

def xyY_to_xyz(xyY):
    """Chuyển đổi xyY sang XYZ."""
    x, y, Y = xyY[..., 0], xyY[..., 1], xyY[..., 2]
    # Xử lý chia cho 0 (không dùng EPSILON)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Tính X
        X = np.divide(x * Y, y)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) # Thay NaN/Inf bằng 0
        # Tính Z
        Z = np.divide((1 - x - y) * Y, y)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0) # Thay NaN/Inf bằng 0
    return np.stack((X, Y, Z), axis=-1)

# --- Hàm lọc xyY (Phương pháp của Paper - Không lọc Y) ---
def xyY_filter_paper(xyy_image, sigma=2, filter_x=True, filter_y=True):
    """
    Áp dụng bộ lọc xyY theo phương pháp mô tả trong paper (Mục 2.3.3 và 2.4).
    Kênh Y được giữ nguyên không lọc.
    """
    x_f, y_f, Y_f = xyy_image[..., 0], xyy_image[..., 1], xyy_image[..., 2]

    # Tính trọng số w = Yf / yf (xử lý chia cho 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.divide(Y_f, y_f)
        w = np.nan_to_num(w)

    # Tính mẫu số chung h*(Yf/yf)
    denominator_x_y = gaussian_filter(w, sigma=sigma)

    # Lọc kênh x (nếu được yêu cầu)
    if filter_x:
        # Tính tử số: h * (xf * w) = h * (xf * Yf / yf)
        numerator_x = gaussian_filter(x_f * w, sigma=sigma)
        # Tính x_g (xử lý chia cho 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            x_g = np.divide(numerator_x, denominator_x_y)
            x_g = np.nan_to_num(x_g)
    else:
        x_g = x_f # Giữ nguyên x

    # Lọc kênh y (nếu được yêu cầu)
    if filter_y:
        # Tính tử số: h * Yf (Y lọc tạm thời)
        Y_temp_filtered = gaussian_filter(Y_f, sigma=sigma)
        # Tính y_g (xử lý chia cho 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            y_g = np.divide(Y_temp_filtered, denominator_x_y)
            y_g = np.nan_to_num(y_g)
    else:
        y_g = y_f # Giữ nguyên y

    # Giữ nguyên kênh Y gốc theo yêu cầu thực nghiệm
    Y_g = Y_f

    return np.stack((x_g, y_g, Y_g), axis=-1)

# --- Hàm lọc "ngây thơ" (Naive Filter) ---
def naive_filter(xyy_image, sigma=2, filter_x=True, filter_y=True):
    """
    Áp dụng bộ lọc Gaussian trực tiếp lên kênh x và y (phương pháp ngây thơ).
    Kênh Y được giữ nguyên không lọc để so sánh.
    """
    x_f, y_f, Y_f = xyy_image[..., 0], xyy_image[..., 1], xyy_image[..., 2]
    if filter_x:
        x_g = gaussian_filter(x_f, sigma=sigma)
    else:
        x_g = x_f
    if filter_y:
        y_g = gaussian_filter(y_f, sigma=sigma)
    else:
        y_g = y_f
    Y_g = Y_f
    return np.stack((x_g, y_g, Y_g), axis=-1)

# --- Hàm tính Rho (Năng lượng tương đối) ---
def calculate_relative_energy(rgb_image_uint8):
    """Tính [rho_R, rho_G, rho_B] cho ảnh RGB uint8."""
    if rgb_image_uint8.ndim != 3 or rgb_image_uint8.shape[2] != 3:
        raise ValueError("Đầu vào phải là ảnh RGB HxWx3.")
    if rgb_image_uint8.dtype != np.uint8:
         raise ValueError("Kiểu dữ liệu đầu vào phải là uint8.")

    rgb_image_float = rgb_image_uint8.astype(np.float64) / 255.0
    energies = []
    for i in range(3): # Duyệt qua R, G, B
        channel = rgb_image_float[..., i]
        fft_channel = np.fft.fft2(channel)
        energy = np.sum(np.abs(fft_channel)**2)
        energies.append(energy)

    total_energy = np.sum(energies)
    # Xử lý chia cho 0 nếu ảnh đen hoàn toàn
    if total_energy == 0:
        return [0.0, 0.0, 0.0]
    relative_energies = [e / total_energy for e in energies]
    return relative_energies

# --- Hàm vẽ biểu đồ sắc độ xy ---
def plot_chromaticity_diagram(ax, x_coords, y_coords, title):
    """Vẽ biểu đồ phân tán xy trên một trục (Axes) của Matplotlib."""
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    # Bỏ qua các giá trị không hợp lệ (NaN, Inf) khi vẽ
    valid_indices = np.isfinite(x_flat) & np.isfinite(y_flat)
    ax.scatter(x_flat[valid_indices], y_flat[valid_indices], s=1, alpha=0.1, marker='.') # s=kích thước điểm, alpha=độ trong suốt
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_aspect('equal', adjustable='box') # Giữ tỉ lệ trục bằng nhau

# --- Main ---
if __name__ == "__main__":
    # --- Cấu hình ---
    input_filename = "image copy.png" # Ảnh gốc từ report
    output_dir = "filtering_results_final" # Thư mục lưu kết quả
    sigma_paper = 2.0       # Sigma cho phương pháp Paper
    sigma_naive = 10.0      # Sigma LỚN HƠN cho phương pháp Naive để thấy rõ dịch chuyển màu
    # ---

    os.makedirs(output_dir, exist_ok=True)
    print(f"Cảnh báo: Code đang chạy mà không có EPSILON bảo vệ chia cho 0.")

    # --- Đọc ảnh đầu vào ---
    try:
        img_bgr = cv2.imread(input_filename)
        if img_bgr is None:
            raise FileNotFoundError(f"Không thể đọc file ảnh: {input_filename}.")
        img_rgb_matrix = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print(f"Đã đọc ảnh: '{input_filename}', kích thước: {img_rgb_matrix.shape}")
    except FileNotFoundError as e:
        print(e); exit()
    except Exception as e:
        print(f"Lỗi khi đọc ảnh: {e}"); exit()

    # --- Chuẩn bị dữ liệu xyY gốc ---
    print("Chuẩn bị dữ liệu xyY gốc...")
    rgb_float_orig = img_rgb_matrix.astype(np.float64) / 255.0
    xyz_orig = rgb_to_xyz(rgb_float_orig)
    xyy_orig = xyz_to_xyY(xyz_orig)

    # --- Phần 1: Lọc theo Phương pháp Paper (Hình 2 & 3) ---
    print(f"\n--- Phần 1: LỌC THEO PAPER (sigma={sigma_paper}) ---")
    xyy_paper_x = xyY_filter_paper(xyy_orig, sigma=sigma_paper, filter_x=True, filter_y=False)
    xyy_paper_y = xyY_filter_paper(xyy_orig, sigma=sigma_paper, filter_x=False, filter_y=True)
    xyy_paper_xy = xyY_filter_paper(xyy_orig, sigma=sigma_paper, filter_x=True, filter_y=True)

    print("Chuyển đổi kết quả Paper về RGB...")
    rgb_paper_x = (xyz_to_rgb(xyY_to_xyz(xyy_paper_x)) * 255).astype(np.uint8)
    rgb_paper_y = (xyz_to_rgb(xyY_to_xyz(xyy_paper_y)) * 255).astype(np.uint8)
    rgb_paper_xy = (xyz_to_rgb(xyY_to_xyz(xyy_paper_xy)) * 255).astype(np.uint8)

    print("Lưu và hiển thị ảnh kết quả Paper...")
    cv2.imwrite(os.path.join(output_dir, f"output_paper_x_s{sigma_paper}.png"), cv2.cvtColor(rgb_paper_x, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"output_paper_y_s{sigma_paper}.png"), cv2.cvtColor(rgb_paper_y, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"output_paper_xy_s{sigma_paper}.png"), cv2.cvtColor(rgb_paper_xy, cv2.COLOR_RGB2BGR))

    fig_paper_imgs, axes_paper_imgs = plt.subplots(2, 2, figsize=(10, 10))
    # ... (code vẽ ảnh paper như trước) ...
    axes_paper_imgs[0, 0].imshow(img_rgb_matrix)
    axes_paper_imgs[0, 0].set_title("Original Image")
    axes_paper_imgs[0, 0].axis('off')
    axes_paper_imgs[0, 1].imshow(rgb_paper_x)
    axes_paper_imgs[0, 1].set_title(f"x-filtered (Paper, s={sigma_paper})")
    axes_paper_imgs[0, 1].axis('off')
    axes_paper_imgs[1, 0].imshow(rgb_paper_y)
    axes_paper_imgs[1, 0].set_title(f"y-filtered (Paper, s={sigma_paper})")
    axes_paper_imgs[1, 0].axis('off')
    axes_paper_imgs[1, 1].imshow(rgb_paper_xy)
    axes_paper_imgs[1, 1].set_title(f"xy-filtered (Paper, s={sigma_paper})")
    axes_paper_imgs[1, 1].axis('off')
    fig_paper_imgs.suptitle(f'Paper Method Results (Y unfiltered)', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)

    print("\nTính Rho values cho phương pháp Paper...")
    rho_original = calculate_relative_energy(img_rgb_matrix)
    rho_paper_x = calculate_relative_energy(rgb_paper_x)
    rho_paper_y = calculate_relative_energy(rgb_paper_y)
    rho_paper_xy = calculate_relative_energy(rgb_paper_xy)
    print(f"Original:      rho_R={rho_original[0]:.4f}, rho_G={rho_original[1]:.4f}, rho_B={rho_original[2]:.4f}")
    print(f"x-filtered (P): rho_R={rho_paper_x[0]:.4f}, rho_G={rho_paper_x[1]:.4f}, rho_B={rho_paper_x[2]:.4f}")
    print(f"y-filtered (P): rho_R={rho_paper_y[0]:.4f}, rho_G={rho_paper_y[1]:.4f}, rho_B={rho_paper_y[2]:.4f}")
    print(f"xy-filtered (P):rho_R={rho_paper_xy[0]:.4f}, rho_G={rho_paper_xy[1]:.4f}, rho_B={rho_paper_xy[2]:.4f}")

    print("Vẽ biểu đồ sắc độ Paper...")
    fig_paper_chroma, axes_paper_chroma = plt.subplots(1, 4, figsize=(18, 5))
    # ... (code vẽ biểu đồ paper như trước) ...
    plot_chromaticity_diagram(axes_paper_chroma[0], xyy_orig[..., 0], xyy_orig[..., 1], "Original Image")
    plot_chromaticity_diagram(axes_paper_chroma[1], xyy_paper_x[..., 0], xyy_paper_x[..., 1], f"x-filtered (Paper, s={sigma_paper})")
    plot_chromaticity_diagram(axes_paper_chroma[2], xyy_paper_y[..., 0], xyy_paper_y[..., 1], f"y-filtered (Paper, s={sigma_paper})")
    plot_chromaticity_diagram(axes_paper_chroma[3], xyy_paper_xy[..., 0], xyy_paper_xy[..., 1], f"xy-filtered (Paper, s={sigma_paper})")
    fig_paper_chroma.suptitle(f'xy Chromaticity Diagrams (Paper Method)', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)
    paper_chroma_filename = f"chromaticity_paper_s{sigma_paper}.png"
    fig_paper_chroma.savefig(os.path.join(output_dir, paper_chroma_filename))
    print(f"Đã lưu biểu đồ sắc độ Paper: {os.path.join(output_dir, paper_chroma_filename)}")

    # --- Phần 2: Lọc Naive (Hình 4 & 5) ---
    print(f"\n--- Phần 2: LỌC THEO PHƯƠNG PHÁP NAIVE (sigma={sigma_naive}) ---")
    xyy_naive_x = naive_filter(xyy_orig, sigma=sigma_naive, filter_x=True, filter_y=False)
    xyy_naive_y = naive_filter(xyy_orig, sigma=sigma_naive, filter_x=False, filter_y=True)
    xyy_naive_xy = naive_filter(xyy_orig, sigma=sigma_naive, filter_x=True, filter_y=True)

    print("Chuyển đổi kết quả Naive về RGB...")
    rgb_naive_x = (xyz_to_rgb(xyY_to_xyz(xyy_naive_x)) * 255).astype(np.uint8)
    rgb_naive_y = (xyz_to_rgb(xyY_to_xyz(xyy_naive_y)) * 255).astype(np.uint8)
    rgb_naive_xy = (xyz_to_rgb(xyY_to_xyz(xyy_naive_xy)) * 255).astype(np.uint8)

    print("Lưu và hiển thị ảnh kết quả Naive...")
    cv2.imwrite(os.path.join(output_dir, f"output_naive_x_s{sigma_naive}.png"), cv2.cvtColor(rgb_naive_x, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"output_naive_y_s{sigma_naive}.png"), cv2.cvtColor(rgb_naive_y, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"output_naive_xy_s{sigma_naive}.png"), cv2.cvtColor(rgb_naive_xy, cv2.COLOR_RGB2BGR))

    fig_naive_imgs, axes_naive_imgs = plt.subplots(2, 2, figsize=(10, 10))
    # ... (code vẽ ảnh naive như trước) ...
    axes_naive_imgs[0, 0].imshow(img_rgb_matrix)
    axes_naive_imgs[0, 0].set_title("Original Image")
    axes_naive_imgs[0, 0].axis('off')
    axes_naive_imgs[0, 1].imshow(rgb_naive_x)
    axes_naive_imgs[0, 1].set_title(f"x-filtered (naive, s={sigma_naive})")
    axes_naive_imgs[0, 1].axis('off')
    axes_naive_imgs[1, 0].imshow(rgb_naive_y)
    axes_naive_imgs[1, 0].set_title(f"y-filtered (naive, s={sigma_naive})")
    axes_naive_imgs[1, 0].axis('off')
    axes_naive_imgs[1, 1].imshow(rgb_naive_xy)
    axes_naive_imgs[1, 1].set_title(f"xy-filtered (naive, s={sigma_naive})")
    axes_naive_imgs[1, 1].axis('off')
    fig_naive_imgs.suptitle(f'Naive Method Filtering Results', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)

    print("\nTính Rho values cho phương pháp Naive...")
    # rho_original đã tính
    rho_naive_x = calculate_relative_energy(rgb_naive_x)
    rho_naive_y = calculate_relative_energy(rgb_naive_y)
    rho_naive_xy = calculate_relative_energy(rgb_naive_xy)
    print(f"Original:      rho_R={rho_original[0]:.4f}, rho_G={rho_original[1]:.4f}, rho_B={rho_original[2]:.4f}")
    print(f"x-filtered (N): rho_R={rho_naive_x[0]:.4f}, rho_G={rho_naive_x[1]:.4f}, rho_B={rho_naive_x[2]:.4f}")
    print(f"y-filtered (N): rho_R={rho_naive_y[0]:.4f}, rho_G={rho_naive_y[1]:.4f}, rho_B={rho_naive_y[2]:.4f}")
    print(f"xy-filtered (N):rho_R={rho_naive_xy[0]:.4f}, rho_G={rho_naive_xy[1]:.4f}, rho_B={rho_naive_xy[2]:.4f}")

    print("Vẽ biểu đồ sắc độ Naive...")
    fig_naive_chroma, axes_naive_chroma = plt.subplots(1, 4, figsize=(18, 5))
    # ... (code vẽ biểu đồ naive như trước) ...
    plot_chromaticity_diagram(axes_naive_chroma[0], xyy_orig[..., 0], xyy_orig[..., 1], "Original Image")
    plot_chromaticity_diagram(axes_naive_chroma[1], xyy_naive_x[..., 0], xyy_naive_x[..., 1], f"x-filtered (naive, s={sigma_naive})")
    plot_chromaticity_diagram(axes_naive_chroma[2], xyy_naive_y[..., 0], xyy_naive_y[..., 1], f"y-filtered (naive, s={sigma_naive})")
    plot_chromaticity_diagram(axes_naive_chroma[3], xyy_naive_xy[..., 0], xyy_naive_xy[..., 1], f"xy-filtered (naive, s={sigma_naive})")
    fig_naive_chroma.suptitle(f'xy Chromaticity Diagrams (Naive Method)', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)
    naive_chroma_filename = f"chromaticity_naive_s{sigma_naive}.png"
    fig_naive_chroma.savefig(os.path.join(output_dir, naive_chroma_filename))
    print(f"Đã lưu biểu đồ sắc độ Naive: {os.path.join(output_dir, naive_chroma_filename)}")

    # --- Kết thúc ---
    print("\nĐóng tất cả cửa sổ hình ảnh để kết thúc chương trình.")
    plt.show() # Giữ các cửa sổ plot mở

    print("\nProcessing complete.")