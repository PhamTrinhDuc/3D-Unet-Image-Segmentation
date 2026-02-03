"""
Mamba SSM - Examples and Usage
Các ví dụ sử dụng Mamba cho different tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, S6, MambaBlock


# ============================================================================
# Example 1: Language Modeling
# ============================================================================
def example_language_modeling():
    """
    Ví dụ: Sử dụng Mamba cho language modeling task
    """
    print("\n" + "="*70)
    print("Example 1: Language Modeling")
    print("="*70)
    
    # Hyperparameters
    vocab_size = 10000
    d_model = 512
    n_layers = 8
    batch_size = 4
    seq_len = 256
    
    # Create model
    model = Mamba(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
        d_state=16,
        d_conv=4,
        expand=2
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Sample input (token ids)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)  # (batch, seq_len, vocab_size)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Compute loss (example)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )
    print(f"Cross-entropy loss: {loss.item():.4f}")


# ============================================================================
# Example 2: Sequence Classification
# ============================================================================
def example_sequence_classification():
    """
    Ví dụ: Sử dụng Mamba cho sequence classification (sentiment analysis, etc.)
    """
    print("\n" + "="*70)
    print("Example 2: Sequence Classification")
    print("="*70)
    
    class MambaClassifier(nn.Module):
        def __init__(self, vocab_size, d_model, n_layers, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.mamba = Mamba(
                d_model=d_model,
                n_layers=n_layers,
                d_state=16,
                d_conv=4,
                expand=2
            )
            self.classifier = nn.Linear(d_model, num_classes)
        
        def forward(self, x):
            # x: (batch, seq_len)
            x = self.embedding(x)  # (batch, seq_len, d_model)
            x = self.mamba(x)      # (batch, seq_len, d_model)
            # Pool: take last token representation
            x = x[:, -1, :]        # (batch, d_model)
            logits = self.classifier(x)  # (batch, num_classes)
            return logits
    
    # Hyperparameters
    vocab_size = 5000
    d_model = 256
    n_layers = 4
    num_classes = 3  # e.g., negative, neutral, positive
    batch_size = 8
    seq_len = 128
    
    model = MambaClassifier(vocab_size, d_model, n_layers, num_classes)
    print(f"Classifier created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Sample data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Compute accuracy
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean()
    print(f"Random accuracy: {accuracy.item():.2%}")


# ============================================================================
# Example 3: Text Generation
# ============================================================================
def example_text_generation():
    """
    Ví dụ: Generate text using Mamba (greedy decoding)
    """
    print("\n" + "="*70)
    print("Example 3: Text Generation")
    print("="*70)
    
    # Simple vocabulary for demo
    vocab = {
        '<pad>': 0, '<bos>': 1, '<eos>': 2,
        'hello': 3, 'world': 4, 'how': 5, 'are': 6, 'you': 7,
        'I': 8, 'am': 9, 'fine': 10, 'thanks': 11, '.': 12
    }
    idx_to_token = {v: k for k, v in vocab.items()}
    
    vocab_size = len(vocab)
    d_model = 128
    n_layers = 2
    
    # Create model
    model = Mamba(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
        d_state=8,
        d_conv=4,
        expand=2
    )
    
    def generate(model, start_tokens, max_length=20, temperature=1.0):
        """
        Generate sequence using greedy decoding
        """
        model.eval()
        generated = start_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = model(generated)  # (1, seq_len, vocab_size)
                
                # Get logits for last position
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sample (greedy)
                next_token = torch.argmax(next_token_logits)
                
                # Append to sequence
                generated = torch.cat([
                    generated,
                    next_token.unsqueeze(0).unsqueeze(0)
                ], dim=1)
                
                # Stop if EOS
                if next_token.item() == vocab['<eos>']:
                    break
        
        return generated
    
    # Start with "<bos> hello"
    start = torch.tensor([[vocab['<bos>'], vocab['hello']]])
    
    print(f"Starting sequence: {[idx_to_token[i.item()] for i in start[0]]}")
    
    # Generate (note: model is random, output will be meaningless)
    generated = generate(model, start, max_length=10)
    
    generated_text = [idx_to_token.get(i.item(), '<unk>') for i in generated[0]]
    print(f"Generated sequence: {generated_text}")
    print("Note: Model is untrained, so output is random!")


# ============================================================================
# Example 4: Understanding State Evolution
# ============================================================================
def example_state_evolution():
    """
    Ví dụ: Visualize how hidden state evolves through sequence
    """
    print("\n" + "="*70)
    print("Example 4: Hidden State Evolution")
    print("="*70)
    
    d_model = 64
    seq_len = 10
    
    s6 = S6(d_model=d_model, d_state=8, d_conv=4, expand=2)
    
    # Create simple input
    x = torch.randn(1, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"d_inner: {s6.d_inner}")
    print(f"d_state: {s6.d_state}")
    
    # Track state evolution (simplified version)
    with torch.no_grad():
        # Get projected input
        x_and_z = s6.in_proj(x)
        x_proj, z = x_and_z.split([s6.d_inner, s6.d_inner], dim=-1)
        
        # Conv and activation
        x_proj = torch.nn.functional.silu(x_proj)
        
        print(f"\nAfter projection and activation: {x_proj.shape}")
        
        # Compute norms at each timestep
        norms = torch.norm(x_proj[0], dim=-1)
        print(f"\nNorms at each position:")
        for i, norm in enumerate(norms):
            print(f"  Position {i}: {norm.item():.4f}")


# ============================================================================
# Example 5: Comparing with baseline (MLP)
# ============================================================================
def example_comparison():
    """
    So sánh Mamba với một baseline đơn giản
    """
    print("\n" + "="*70)
    print("Example 5: Comparison with MLP baseline")
    print("="*70)
    
    class MLPModel(nn.Module):
        def __init__(self, vocab_size, d_model, seq_len):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.flatten_dim = seq_len * d_model
            self.mlp = nn.Sequential(
                nn.Linear(self.flatten_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, vocab_size * seq_len)
            )
            self.seq_len = seq_len
            self.vocab_size = vocab_size
        
        def forward(self, x):
            x = self.embedding(x)  # (batch, seq_len, d_model)
            x = x.reshape(x.size(0), -1)  # flatten
            x = self.mlp(x)
            x = x.reshape(x.size(0), self.seq_len, self.vocab_size)
            return x
    
    vocab_size = 1000
    d_model = 128
    seq_len = 64
    n_layers = 2
    
    # Create both models
    mamba_model = Mamba(d_model, n_layers, vocab_size)
    mlp_model = MLPModel(vocab_size, d_model, seq_len)
    
    # Count parameters
    mamba_params = sum(p.numel() for p in mamba_model.parameters())
    mlp_params = sum(p.numel() for p in mlp_model.parameters())
    
    print(f"Mamba parameters: {mamba_params:,}")
    print(f"MLP parameters: {mlp_params:,}")
    print(f"Ratio: {mlp_params / mamba_params:.2f}x")
    
    # Test inference time (rough estimate)
    import time
    
    x = torch.randint(0, vocab_size, (1, seq_len))
    
    # Warmup
    with torch.no_grad():
        _ = mamba_model(x)
        _ = mlp_model(x)
    
    # Mamba
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = mamba_model(x)
    mamba_time = (time.time() - start) / 10
    
    # MLP
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = mlp_model(x)
    mlp_time = (time.time() - start) / 10
    
    print(f"\nAverage inference time (10 runs):")
    print(f"  Mamba: {mamba_time*1000:.2f} ms")
    print(f"  MLP: {mlp_time*1000:.2f} ms")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MAMBA SSM - USAGE EXAMPLES")
    print("="*70)
    
    try:
        example_language_modeling()
    except Exception as e:
        print(f"Error in Example 1: {e}")
    
    try:
        example_sequence_classification()
    except Exception as e:
        print(f"Error in Example 2: {e}")
    
    try:
        example_text_generation()
    except Exception as e:
        print(f"Error in Example 3: {e}")
    
    try:
        example_state_evolution()
    except Exception as e:
        print(f"Error in Example 4: {e}")
    
    try:
        example_comparison()
    except Exception as e:
        print(f"Error in Example 5: {e}")
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)