import torch
import torch.nn as nn
from src.tagan.layers.temporal_attention import AsymmetricTemporalAttention

def test_attention_mask_handling():
    """Test the attention mask handling in AsymmetricTemporalAttention."""
    print("\n=== Testing Attention Mask Handling ===")
    
    # Create sample data
    batch_size = 2
    seq_len = 5
    hidden_dim = 16
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Create time stamps
    time_stamps = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    
    # Create attention mask (fully attending)
    attention_mask = torch.ones(batch_size, seq_len, seq_len)
    
    # Create attention module
    attention = AsymmetricTemporalAttention(
        hidden_dim=hidden_dim,
        num_heads=4,
        time_aware=True
    )
    
    try:
        # Test with tensor mask
        print("Testing with tensor mask...")
        output = attention(x, time_stamps=time_stamps, attention_mask=attention_mask)
        print(f"Output shape with tensor mask: {output.shape}")
        
        # Test with list mask
        print("Testing with list mask...")
        mask_list = [torch.ones(seq_len, seq_len) for _ in range(seq_len)]
        output = attention(x, time_stamps=time_stamps, attention_mask=mask_list)
        print(f"Output shape with list mask: {output.shape}")
        
        # Test with mismatched mask dimensions
        print("Testing with mismatched mask dimensions...")
        wrong_mask = torch.ones(batch_size, seq_len + 2, seq_len + 2)
        output = attention(x, time_stamps=time_stamps, attention_mask=wrong_mask)
        print(f"Output shape with wrong mask dimensions: {output.shape}")
        
        print("All attention mask tests passed!")
    except Exception as e:
        print(f"Error in attention mask handling: {str(e)}")
        raise e

def test_time_based_attention():
    """Test the time-based attention calculations."""
    print("\n=== Testing Time-Based Attention ===")
    
    # Create sample data
    batch_size = 2
    seq_len = 5
    hidden_dim = 16
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Create different time stamps patterns
    # 1. Uniform time stamps
    uniform_time = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    
    # 2. Non-uniform time stamps
    non_uniform_time = torch.tensor([[0.0, 1.0, 3.0, 8.0, 10.0], 
                                     [0.0, 2.0, 5.0, 9.0, 15.0]], dtype=torch.float)
    
    # 3. Very large time differences
    large_diff_time = torch.tensor([[0.0, 100.0, 1000.0, 5000.0, 10000.0], 
                                    [0.0, 50.0, 500.0, 2000.0, 8000.0]], dtype=torch.float)
    
    # Create attention module
    attention = AsymmetricTemporalAttention(
        hidden_dim=hidden_dim,
        num_heads=4,
        time_aware=True
    )
    
    try:
        # Test with uniform time
        print("Testing with uniform time stamps...")
        output = attention(x, time_stamps=uniform_time)
        print(f"Output shape with uniform time: {output.shape}")
        
        # Test with non-uniform time
        print("Testing with non-uniform time stamps...")
        output = attention(x, time_stamps=non_uniform_time)
        print(f"Output shape with non-uniform time: {output.shape}")
        
        # Test with large time differences
        print("Testing with large time differences...")
        output = attention(x, time_stamps=large_diff_time)
        print(f"Output shape with large time differences: {output.shape}")
        
        print("All time-based attention tests passed!")
    except Exception as e:
        print(f"Error in time-based attention: {str(e)}")
        raise e

def test_multi_scale_attention():
    """Test the multi-scale attention mechanism."""
    from src.tagan.layers.temporal_attention import MultiTimeScaleAttention
    print("\n=== Testing Multi-Scale Attention ===")
    
    # Create sample data
    batch_size = 2
    seq_len = 7  # Prime number to test non-divisible sequence lengths
    hidden_dim = 16
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Create time stamps
    time_stamps = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    
    # Create attention module
    attention = MultiTimeScaleAttention(
        hidden_dim=hidden_dim,
        num_scales=3,
        scale_factors=[1, 2, 3],  # Use factors that don't divide seq_len evenly
        num_heads=4,
        time_aware=True
    )
    
    try:
        # Test basic forward pass
        print("Testing multi-scale attention...")
        output = attention(x, time_stamps=time_stamps)
        print(f"Output shape: {output.shape}")
        
        # Test with very short sequence
        print("Testing with very short sequence...")
        short_x = torch.randn(batch_size, 2, hidden_dim)
        short_time = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float)
        output = attention(short_x, time_stamps=short_time)
        print(f"Output shape with short sequence: {output.shape}")
        
        print("All multi-scale attention tests passed!")
    except Exception as e:
        print(f"Error in multi-scale attention: {str(e)}")
        raise e

if __name__ == "__main__":
    print("Running diagnostic tests for TAGAN attention mechanisms...")
    
    try:
        test_attention_mask_handling()
        test_time_based_attention()
        test_multi_scale_attention()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()