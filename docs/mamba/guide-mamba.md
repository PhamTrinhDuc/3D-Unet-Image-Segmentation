# Mamba SSM - Hướng dẫn chi tiết

## 1. Giới thiệu về State Space Models (SSM)

State Space Models mô tả hệ thống động với hai phương trình:

```
h'(t) = Ah(t) + Bx(t)    # State equation (continuous time)
y(t) = Ch(t) + Dx(t)     # Output equation
```

Trong đó:
- `x(t)`: input tại thời điểm t
- `h(t)`: hidden state (vector N chiều)
- `y(t)`: output
- `A`: ma trận dynamics (N×N)
- `B`: ma trận input-to-state (N×1)
- `C`: ma trận state-to-output (1×N)
- `D`: skip connection (scalar)

## 2. Discretization (Rời rạc hóa)

Để sử dụng với dữ liệu rời rạc, ta cần discretize SSM:

**Zero-Order Hold (ZOH):**
```
h_t = Ā·h_{t-1} + B̄·x_t
y_t = C·h_t + D·x_t

Với:
Ā = exp(Δ·A)
B̄ = (Ā - I)·A^{-1}·B ≈ Δ·B
```

## 3. Selective SSM (Mamba)

**Điểm khác biệt chính:**
- Mamba làm cho B, C, và Δ (time-step) **phụ thuộc vào input**
- Điều này cho phép model "chọn lọc" thông tin quan trọng

**Công thức:**
```python
B_t = Linear_B(x_t)     # Content-aware
C_t = Linear_C(x_t)     # Content-aware  
Δ_t = Softplus(Linear_Δ(x_t))  # Content-aware time-step
```

## 4. Kiến trúc Mamba Block

```
Input x
   ↓
Linear projection → [x_branch, z_branch]
   ↓                          ↓
Conv1D (causal)              |
   ↓                          |
SiLU                         |
   ↓                          |
Project → [Δ, B, C]          |
   ↓                          |
SSM (Selective Scan)         |
   ↓                          |
   ↓ ←─── Gating ←───────────┘
   ↓    (element-wise *)
Linear projection
   ↓
Output
```

## 5. Ưu điểm của Mamba

1. **Efficient long sequences**: O(N) complexity thay vì O(N²) của Transformer
2. **Selective information**: Có thể học được thông tin nào cần nhớ
3. **Hardware-efficient**: Scan operation tối ưu cho GPU
4. **No attention**: Không cần attention mechanism

## 6. So sánh với Transformer

| Aspect | Transformer | Mamba |
|--------|-------------|-------|
| Complexity | O(N²) | O(N) |
| Memory | O(N²) | O(N) |
| Long sequences | Khó | Dễ |
| Parallelization | Tốt | Moderate |
| Inductive bias | Ít | SSM structure |

## 7. Ví dụ sử dụng

### Ví dụ 1: Sequence modeling cơ bản
```python
import torch
from mamba_ssm import Mamba

# Khởi tạo model
model = Mamba(
    d_model=256,      # Dimension của embedding
    n_layers=4,       # Số layers
    d_state=16,       # Dimension của state space
    expand_factor=2,  # Expand dimension trong block
)

# Input: (batch, seq_len, d_model)
x = torch.randn(2, 100, 256)

# Forward pass
output = model(x)  # (2, 100, 256)
```

### Ví dụ 2: Language Modeling
```python
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mamba = Mamba(d_model, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.mamba(x)               # (batch, seq_len, d_model)
        logits = self.lm_head(x)        # (batch, seq_len, vocab_size)
        return logits

# Sử dụng
vocab_size = 10000
model = MambaLM(vocab_size)

# Giả sử có input_ids
input_ids = torch.randint(0, vocab_size, (2, 50))
logits = model(input_ids)
print(f"Logits shape: {logits.shape}")  # (2, 50, 10000)
```

### Ví dụ 3: Time Series Prediction
```python
import torch
from mamba_ssm import Mamba

class TimeSeriesModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.mamba = Mamba(d_model, n_layers)
        self.output_proj = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.mamba(x)
        x = self.output_proj(x)
        return x

# Dữ liệu time series
input_dim = 5   # 5 features
output_dim = 1  # predict 1 value
model = TimeSeriesModel(input_dim, output_dim)

# Input: 30 timesteps
x = torch.randn(4, 30, input_dim)
predictions = model(x)
print(f"Predictions: {predictions.shape}")  # (4, 30, 1)
```

## 8. Training Tips

```python
import torch.optim as optim

# Setup
model = Mamba(d_model=256, n_layers=4)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        x, labels = batch
        
        # Forward
        output = model(x)
        loss = criterion(output.view(-1, vocab_size), labels.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 9. Tham số quan trọng

- **d_model**: Dimension chính (thường 256-1024)
- **d_state**: State space dimension (16-64), càng lớn càng có khả năng nhớ nhiều
- **expand_factor**: Mở rộng dimension trong block (2-4)
- **n_layers**: Số layers (4-24)
- **dt_rank**: Rank của delta projection (auto = d_model/16)

## 10. Debugging tips

```python
# Check output shape
print(f"Input: {x.shape}")
print(f"Output: {output.shape}")

# Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean():.6f}")
    else:
        print(f"{name}: No gradient!")

# Visualize SSM parameters
import matplotlib.pyplot as plt

# Get A matrix (dynamics)
A = -torch.exp(model.layers[0]['mamba'].ssm.A_log)
plt.imshow(A.detach().cpu(), aspect='auto')
plt.colorbar()
plt.title("SSM Dynamics Matrix A")
plt.show()
```

## Tài liệu tham khảo

- Paper: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- GitHub: https://github.com/state-spaces/mamba