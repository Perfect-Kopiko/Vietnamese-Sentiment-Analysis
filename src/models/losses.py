"""
Module các hàm mất mát (Loss functions) xử lý mất cân bằng lớp (Class Imbalance)
trong bài toán Phân tích cảm xúc văn bản tiếng Việt.
Hỗ trợ:
- CrossEntropy chuẩn ('ce')
- Weighted CrossEntropy ('weighted_ce') với trọng số nghịch đảo tần suất lớp
- Focal Loss đa lớp ('focal') với tham số điều chỉnh tập trung gamma và alpha
"""

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    labels: Union[List[int], np.ndarray, torch.Tensor],
    num_classes: int = 3,
    smoothing: str = "sqrt"
) -> torch.Tensor:
    """
    Tính toán trọng số nghịch đảo tần suất lớp để xử lý tình trạng mất cân bằng dữ liệu
    (đặc biệt là lớp Neutral chỉ chiếm khoảng ~4% trong bộ dữ liệu UIT-VSFC).

    Công thức:
    - w_i = (N / (num_classes * count_i))
    - Làm mượt (smoothing='sqrt'): w_i = sqrt(N / count_i)
    - Chuẩn hóa: sum(w_i) = num_classes để giữ độ lớn gradient ổn định.

    Args:
        labels: Danh sách hoặc mảng nhãn của tập huấn luyện.
        num_classes: Số lượng lớp (mặc định 3: Negative, Neutral, Positive).
        smoothing: Kiểu làm mượt ('sqrt', 'none'). 'sqrt' giúp giảm độ chênh lệch cực đoan.

    Returns:
        torch.Tensor: Tensor 1D kích thước (num_classes,) chứa trọng số mỗi lớp kiểu float32.
    """
    if isinstance(labels, torch.Tensor):
        labels_arr = labels.detach().cpu().numpy()
    else:
        labels_arr = np.asarray(labels)

    total_samples = len(labels_arr)
    weights = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        count = np.sum(labels_arr == c)
        if count == 0:
            # Fallback tránh chia cho 0 nếu lớp vắng mặt
            weights[c] = 1.0
        else:
            if smoothing == "sqrt":
                weights[c] = np.sqrt(total_samples / count)
            else:
                weights[c] = total_samples / (num_classes * count)

    # Chuẩn hóa sao cho tổng trọng số bằng num_classes tránh bùng nổ gradient
    sum_w = np.sum(weights)
    if sum_w > 0:
        weights = (weights / sum_w) * num_classes

    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Focal Loss đa lớp (Multi-class Focal Loss) được đề xuất bởi Lin et al. (RetinaNet).
    Công thức:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Trong đó:
        - p_t: Xác suất dự đoán cho nhãn đúng
        - gamma: Hệ số focusing parameter (giảm tổn thất đối với các mẫu dễ phân loại, tập trung vào mẫu khó)
        - alpha: Trọng số của từng lớp để giải quyết mất cân bằng lớp
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        """
        Khởi tạo FocalLoss.

        Args:
            gamma: Hệ số tập trung vào mẫu khó (thường chọn từ 1.0 đến 3.0, mặc định 2.0).
            alpha: Tensor 1D shape (num_classes,) chứa trọng số cân bằng lớp.
            reduction: 'mean', 'sum', hoặc 'none'.
        """
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            # Đảm bảo alpha là float tensor và đăng ký buffer nếu cần
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Tính toán Focal Loss giữa logits và targets.

        Args:
            logits: Tensor dự đoán chưa qua softmax, shape (batch_size, num_classes).
            targets: Tensor nhãn thực tế, shape (batch_size,).

        Returns:
            torch.Tensor: Giá trị loss scalar (nếu reduction='mean' hoặc 'sum')
                          hoặc vector shape (batch_size,) (nếu reduction='none').
        """
        # log_softmax tính toán ổn định về mặt số học
        log_p = F.log_softmax(logits, dim=-1)  # (batch_size, num_classes)
        # Lấy log p_t tương ứng với target của từng mẫu
        log_pt = log_p.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)  # (batch_size,)
        pt = log_pt.exp()  # (batch_size,)

        # Tính toán thành phần focal: (1 - pt)^gamma
        focal_weight = torch.pow(1.0 - pt, self.gamma)

        loss = -focal_weight * log_pt

        # Nhân trọng số alpha nếu có
        if self.alpha is not None:
            alpha_device = self.alpha.to(targets.device)
            alpha_t = alpha_device.gather(dim=0, index=targets)  # (batch_size,)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Không hỗ trợ reduction='{self.reduction}'. Chỉ hỗ trợ 'mean', 'sum', 'none'.")


def get_loss_function(
    loss_type: str = "ce",
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    device: str = "cpu"
) -> nn.Module:
    """
    Hàm Factory khởi tạo hàm mất mát tương ứng theo cấu hình.

    Args:
        loss_type: Loại loss ('ce', 'weighted_ce', 'focal').
        class_weights: Trọng số lớp (Tensor 1D shape (num_classes,)).
        focal_gamma: Tham số gamma cho Focal Loss.
        device: Thiết bị tính toán ('cpu' hoặc 'cuda').

    Returns:
        nn.Module: Đối tượng hàm mất mát sẵn sàng sử dụng.
    """
    loss_type = loss_type.lower()

    if loss_type == "ce":
        print("[*] Sử dụng hàm mất mát: CrossEntropyLoss (tiêu chuẩn).")
        return nn.CrossEntropyLoss().to(device)

    elif loss_type == "weighted_ce":
        if class_weights is not None:
            weights = class_weights.to(device)
            print(f"[*] Sử dụng hàm mất mát: Weighted CrossEntropyLoss (weights={weights.cpu().numpy().round(3)}).")
            return nn.CrossEntropyLoss(weight=weights).to(device)
        else:
            print("[!] Cảnh báo: class_weights=None cho weighted_ce. Fallback sang CrossEntropyLoss chuẩn.")
            return nn.CrossEntropyLoss().to(device)

    elif loss_type == "focal":
        alpha = class_weights.to(device) if class_weights is not None else None
        alpha_str = f", alpha={alpha.cpu().numpy().round(3)}" if alpha is not None else ""
        print(f"[*] Sử dụng hàm mất mát: FocalLoss (gamma={focal_gamma}{alpha_str}).")
        return FocalLoss(gamma=focal_gamma, alpha=alpha, reduction="mean").to(device)

    else:
        raise ValueError(f"Loại loss '{loss_type}' không hợp lệ. Chọn 'ce', 'weighted_ce' hoặc 'focal'.")


if __name__ == "__main__":
    print("=== TEST MODULE losses.py ===")
    # Giả lập phân bố nhãn thực tế của UIT-VSFC (~16% negative (0), ~4% neutral (1), ~80% positive (2))
    simulated_labels = [0] * 160 + [1] * 40 + [2] * 800
    weights = compute_class_weights(simulated_labels, num_classes=3)
    print(f"Class Weights (chuẩn hóa tổng=3): {weights.numpy()}")
    print(f"Tổng trọng số: {weights.sum().item():.4f}")

    # Giả lập logits (batch_size=4, num_classes=3) và targets
    dummy_logits = torch.tensor([
        [2.0, -1.0, 0.5],
        [0.1,  3.0, -0.2],
        [-0.5, 0.0,  2.5],
        [1.0,  0.5, -1.0]
    ], dtype=torch.float32)
    dummy_targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    # 1. Test CE
    ce_loss = get_loss_function("ce")
    l_ce = ce_loss(dummy_logits, dummy_targets)
    print(f"Loss CE: {l_ce.item():.4f}")

    # 2. Test Weighted CE
    wce_loss = get_loss_function("weighted_ce", class_weights=weights)
    l_wce = wce_loss(dummy_logits, dummy_targets)
    print(f"Loss Weighted CE: {l_wce.item():.4f}")

    # 3. Test Focal Loss
    focal_loss = get_loss_function("focal", class_weights=weights, focal_gamma=2.0)
    l_focal = focal_loss(dummy_logits, dummy_targets)
    print(f"Loss Focal (gamma=2.0): {l_focal.item():.4f}")
