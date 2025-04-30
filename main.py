import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def rgb_to_xyz(rgb):
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    shape = rgb.shape
    rgb = rgb.reshape(-1, 3)
    xyz = rgb @ M.T
    return xyz.reshape(shape)

def xyz_to_rgb(xyz):
    M = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    shape = xyz.shape
    xyz = xyz.reshape(-1, 3)
    rgb = xyz @ M.T
    return np.clip(rgb.reshape(shape), 0, 1)

def xyz_to_xyY(xyz):
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    denom = X + Y + Z
    x = X / denom
    y = Y / denom
    return np.stack((x, y, Y), axis=-1)

def xyY_to_xyz(xyY):
    x, y, Y = xyY[..., 0], xyY[..., 1], xyY[..., 2]
    X = (x * Y) / (y )
    Z = ((1 - x - y) * Y) / (y )
    return np.stack((X, Y, Z), axis=-1)

def xyY_filter(image_rgb, sigma=2):
    rgb = image_rgb / 255.0
    xyz = rgb_to_xyz(rgb)
    xyY = xyz_to_xyY(xyz)
    x, y, Y = xyY[..., 0], xyY[..., 1], xyY[..., 2]
    w = Y / (y )

    x_filtered = gaussian_filter(x * w, sigma=sigma) / (gaussian_filter(w, sigma=sigma) )
    Y_filtered = gaussian_filter(Y, sigma=sigma)
    y_filtered = Y_filtered / (gaussian_filter(w, sigma=sigma) )

    xyY_filtered = np.stack((x_filtered, y_filtered, Y_filtered), axis=-1)
    xyz_filtered = xyY_to_xyz(xyY_filtered)
    rgb_filtered = xyz_to_rgb(xyz_filtered)
    return (rgb_filtered * 255).astype(np.uint8)

if __name__ == "__main__":
    # Đọc ảnh và chuyển sang RGB
    img = cv2.imread("image.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Lọc trong không gian xyY
    filtered = xyY_filter(img, sigma=2)

    # Hiển thị
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(filtered)
    plt.title("xy-filtered")
    plt.axis("off")

    plt.show()

    # —— Phần lưu ảnh kết quả —— #
    # Lưu ảnh lọc dưới định dạng PNG
    # Chuyển ngược sang BGR để dùng cv2.imwrite
    out_bgr = cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_image.png", out_bgr)
    print("Da luu anh ket qua: output_image.png")
