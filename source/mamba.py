"""
Mamba SSM (State Space Model) Implementation
Đây là implementation đơn giản của Selective State Space Model trong Mamba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class S6(nn.Module):
    """
    S6: Selective State Space Model - Core của Mamba
    
    SSM được định nghĩa bởi các phương trình:
    h'(t) = A h(t) + B x(t)  # continuous time
    y(t) = C h(t)
    
    Sau khi discretize:
    h_t = A_bar h_{t-1} + B_bar x_t
    y_t = C h_t
    """
    
    def __init__(
        self,
        d_model,      # Dimension của input
        d_state=16,   # Dimension của state space (N)
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Tính dt_rank (rank của delta projection)
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank
            
        # Learnable parameters
        # A: (d_model, d_state) - dynamics matrix
        # Khởi tạo A với giá trị âm để stable
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Log space để đảm bảo A âm
        
        # D: (d_model,) - skip connection
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Projection cho delta (time step)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        
        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            
        # Inverse của softplus để khởi tạo bias
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_min)
        
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
    
    def forward(self, x, delta, B, C):
        """
        x: (batch, seq_len, d_model)
        delta: (batch, seq_len, dt_rank)
        B: (batch, seq_len, d_state)
        C: (batch, seq_len, d_state)
        
        Returns:
        y: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Get A from log space
        A = -torch.exp(self.A_log)  # (d_model, d_state)
        
        # Project delta
        delta = self.dt_proj(delta)  # (batch, seq_len, d_model)
        delta = F.softplus(delta)
        
        # Discretization: Zero-order hold (ZOH)
        # A_bar = exp(delta * A)
        # B_bar = (A_bar - I) * A^{-1} * B ≈ delta * B (approximation)
        
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (batch, seq_len, d_model, d_state)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq_len, d_model, d_state)
        
        # Scan: tính h_t = A_bar * h_{t-1} + B_bar * x_t
        h = torch.zeros(batch, d_model, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)
            y = torch.einsum('bdn,bn->bd', h, C[:, t])
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (batch, seq_len, d_model)
        
        # Skip connection
        y = y + self.D * x
        
        return y


class MambaBlock(nn.Module):
    """
    Một block của Mamba architecture
    """
    
    def __init__(
        self,
        d_model,
        d_state=16,
        expand_factor=2,
        dt_rank="auto",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_factor
        self.d_state = d_state
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution (causal)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=4,
            padding=3,
            groups=self.d_inner,
        )
        
        # SSM parameters projection
        # x_proj projects to [delta, B, C]
        if dt_rank == "auto":
            dt_rank = math.ceil(d_model / 16)
        
        self.x_proj = nn.Linear(
            self.d_inner,
            dt_rank + self.d_state * 2,
            bias=False
        )
        
        self.dt_rank = dt_rank
        
        # SSM
        self.ssm = S6(
            d_model=self.d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
        )
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Input projection (split into x and z for gating)
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # Causal convolution
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # Causal: remove extra padding
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # Activation
        x = F.silu(x)
        
        # SSM parameters projection
        x_proj = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)
        
        delta, B, C = torch.split(
            x_proj,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        
        # SSM
        y = self.ssm(x, delta, B, C)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output


class Mamba(nn.Module):
    """
    Mamba Model - Stack of Mamba blocks
    """
    
    def __init__(
        self,
        d_model,
        n_layers,
        d_state=16,
        expand_factor=2,
        dt_rank="auto",
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'mamba': MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand_factor=expand_factor,
                    dt_rank=dt_rank,
                )
            })
            for _ in range(n_layers)
        ])
        
    def forward(self, x):
      """
      x: (batch, seq_len, d_model)
      """
      for layer in self.layers:
          # Pre-norm + residual
          x = x + layer['mamba'](layer['norm'](x))
      
      return x


# Test code
if __name__ == "__main__":
    print("Testing Mamba SSM Implementation\n")
    
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 64
    d_state = 16
    n_layers = 2
    
    # Create model
    model = Mamba(
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        expand_factor=2,
    )
    
    # Random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print("\n✓ Test passed!")
    
    # Show one SSM block
    print("\n" + "="*50)
    print("Testing single Mamba Block:")
    block = MambaBlock(d_model=d_model, d_state=d_state)
    x_test = torch.randn(1, 5, d_model)
    y_test = block(x_test)
    print(f"Input: {x_test.shape} -> Output: {y_test.shape}")