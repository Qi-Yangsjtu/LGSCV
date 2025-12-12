import torch
from icecream import ic

def morton3D(x, y, z, max_bits = 10):
    N = x.shape[0]
    morton = torch.zeros(N, dtype=torch.long, device=x.device)

    for i in range(max_bits):
        bit_mask = 1 << i
        morton |= ((x & bit_mask) << (2 * i)) | ((y & bit_mask) << (2 * i + 1)) | ((z & bit_mask) << (2 * i + 2))

    return morton

def morton2D(x,y, max_bits = 16):
    morton = torch.zeros_like(x, dtype=torch.long)
    for i in range(max_bits):
        mask = 1 << i
        morton |= ((x & mask) << i) | ((y & mask) << (i + 1))
    return morton

def morton_sort(points, *features, max_bits=22):
    """
    points: (N, 3) float Tensor
    features: 其他对应特征（N, ...），任意数量
    返回排序后的 points 和 features
    """
    # Step 1: 归一化 → 映射到整数范围 [0, 2^max_bits)
    coords = points - points.min(dim=0)[0]
    coords = coords / (coords.max(dim=0)[0] + 1e-8)
    quantized = (coords * (2**max_bits - 1)).long()  # (N, 3)

    # Step 2: 计算 Morton 编码
    morton_codes = morton3D(quantized[:, 0], quantized[:, 1], quantized[:, 2], max_bits= max_bits)

    # Step 3: 排序索引
    sorted_idx = torch.argsort(morton_codes)

    # Step 4: 应用于所有特征
    points_sorted = points[sorted_idx]
    features_sorted = [f[sorted_idx] for f in features]

    return (points_sorted, *features_sorted)

def morton_sort_image(img):
    C, H, W = img.shape
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=img.device),
        torch.arange(W, device=img.device),
        indexing = 'ij'
    )
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    morton_index = morton2D(x_flat, y_flat)
    img_flat = img.view(C, -1)
    sorted_indices = morton_index.argsort()
    img_sorted = img_flat[:, sorted_indices]

    xy_sorted = torch.stack([x_flat, y_flat], dim = 1)[sorted_indices]

    return img_sorted, xy_sorted

