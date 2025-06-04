"""
Test script for Spiking Graph Wavelet Convolution Network (SGWCN)
Verify model architecture and core functionalities
"""

import torch
import torch.nn as nn
import numpy as np
import time
from src.models import SpikingGraphWaveletNet, SGWCNClassifier


def test_basic_functionality():
    """Test basic model functionality"""
    print("Testing basic SGWCN functionality...")
    
    # Create model
    model = SpikingGraphWaveletNet(
        input_dim=3,
        hidden_dims=[32, 64],
        num_classes=10,
        num_time_steps=4,
        k_neighbors=10,
        chebyshev_order=2
    )
    
    # Test input
    batch_size = 2
    num_points = 128
    point_cloud = torch.randn(batch_size, num_points, 3)
    
    print(f"Input shape: {point_cloud.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(point_cloud)
        print(f"Output shape: {logits.shape}")
        print(f"Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    
    # Test with analysis
    analysis = model.forward_with_analysis(point_cloud)
    print(f"Analysis keys: {list(analysis.keys())}")
    print(f"Number of layers: {len(analysis['layer_outputs'])}")
    print(f"Energy consumption: {analysis['energy_consumption']:.2e} J")
    
    print("✓ Basic functionality test passed!\n")


def test_model_statistics():
    """Test model statistics and energy estimation"""
    print("Testing model statistics...")
    
    model = SGWCNClassifier(num_classes=40, num_points=1024)
    
    # Model statistics
    stats = model.get_model_statistics()
    print(f"Total parameters: {stats['total_parameters']:,}")
    print(f"Number of layers: {stats['num_layers']}")
    print(f"Conv layer params: {stats['conv_layer_params']}")
    
    # Energy estimation
    energy_stats = model.estimate_energy_consumption(
        num_samples=100, 
        points_per_sample=1024
    )
    print(f"Estimated energy reduction: {energy_stats['energy_reduction_factor']:.1f}x")
    print(f"Energy per sample: {energy_stats['energy_per_sample_nj']:.3f} nJ")
    
    print("✓ Model statistics test passed!\n")


def test_graph_builder():
    """Test adaptive graph construction"""
    print("Testing graph construction...")
    
    from src.models.graph_builder import SparseGraphBuilder
    
    graph_builder = SparseGraphBuilder(k=20, beta=1.0, lambda_param=1.0)
    
    # Test data
    batch_size = 2
    num_points = 100
    point_cloud = torch.randn(batch_size, num_points, 3)
    
    # Build graph
    edge_indices, edge_attrs, s_local = graph_builder(point_cloud)
    
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Local scales shape: {s_local.shape}")
    print(f"Local scales range: [{s_local.min().item():.3f}, {s_local.max().item():.3f}]")
    
    if isinstance(edge_indices, list):
        total_edges = sum(ei.shape[1] for ei in edge_indices)
        print(f"Total edges: {total_edges}")
        print(f"Edges per batch: {[ei.shape[1] for ei in edge_indices]}")
    else:
        print(f"Edge indices shape: {edge_indices.shape}")
        print(f"Edge attributes shape: {edge_attrs.shape}")
    
    print("✓ Graph construction test passed!\n")


def test_wavelet_conv():
    """Test adaptive graph wavelet convolution"""
    print("Testing wavelet convolution...")
    
    from src.models.wavelet_conv import AdaptiveGraphWaveletConv
    from src.models.graph_builder import SparseGraphBuilder
    
    # Build graph
    graph_builder = SparseGraphBuilder(k=10)
    point_cloud = torch.randn(2, 50, 3)
    edge_indices, edge_attrs, s_local = graph_builder(point_cloud)
    
    # Test convolution
    conv = AdaptiveGraphWaveletConv(F_in=3, F_out=16, K=2)
    
    # Input features (using point coordinates)
    x = point_cloud  # [2, 50, 3]
    
    output = conv(x, edge_indices, edge_attrs, s_local)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Local scales shape: {s_local.shape}")
    
    # Test computational stats
    stats = conv.get_computational_stats(B=2, N=50, num_edges=1000)
    print(f"Total FLOPs: {stats['total_flops']:,}")
    print(f"Parameters: {stats['parameters']:,}")
    
    print("✓ Wavelet convolution test passed!\n")


def test_bipolar_lif():
    """Test bipolar LIF neurons"""
    print("Testing bipolar LIF neurons...")
    
    from src.models.spiking_neurons import BipolarLIFNeuron
    
    # Create neuron
    neuron = BipolarLIFNeuron(
        membrane_dim=32,
        tau_mem=20.0,
        theta_pos=1.0,
        theta_neg=-1.0
    )
    
    # Test input
    batch_size, time_steps, num_nodes, features = 2, 8, 50, 32
    input_current = torch.randn(batch_size, time_steps, num_nodes, features)
    
    # Reset state
    neuron.reset_state(batch_size, num_nodes, input_current.device)
    
    # Forward pass
    spike_output, membrane_potential = neuron(input_current, return_membrane=True)
    
    print(f"Input shape: {input_current.shape}")
    print(f"Spike output shape: {spike_output.shape}")  # Should be [B, T, N, 2*F]
    print(f"Membrane potential shape: {membrane_potential.shape}")
    
    # Test statistics
    stats = neuron.get_neuron_statistics(spike_output, membrane_potential)
    print(f"Positive firing rate: {stats['pos_firing_rate_mean']:.3f}")
    print(f"Negative firing rate: {stats['neg_firing_rate_mean']:.3f}")
    print(f"Spike balance: {stats['spike_balance']:.3f}")
    print(f"Sparsity: {stats['sparsity']:.3f}")
    
    print("✓ Bipolar LIF test passed!\n")


def test_end_to_end():
    """Test end-to-end forward pass with gradients"""
    print("Testing end-to-end forward pass...")
    
    model = SGWCNClassifier(num_classes=10, num_points=256)
    
    # Test data
    batch_size = 2
    point_cloud = torch.randn(batch_size, 256, 3, requires_grad=True)
    targets = torch.randint(0, 10, (batch_size,))
    
    # Forward pass
    logits = model(point_cloud)
    loss = nn.CrossEntropyLoss()(logits, targets)
    
    print(f"Input shape: {point_cloud.shape}")
    print(f"Targets: {targets}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"Gradients computed: {has_gradients}")
    
    if has_gradients:
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        print(f"Gradient norm range: [{min(grad_norms):.2e}, {max(grad_norms):.2e}]")
    
    print("✓ End-to-end test passed!\n")


def benchmark_performance():
    """Benchmark model performance"""
    print("Benchmarking performance...")
    
    model = SGWCNClassifier(num_classes=40, num_points=1024)
    model.eval()
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8]
    
    for batch_size in batch_sizes:
        point_cloud = torch.randn(batch_size, 1024, 3)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(point_cloud)
        
        # Timing
        start_time = time.time()
        num_runs = 10
        
        with torch.no_grad():
            for _ in range(num_runs):
                logits = model(point_cloud)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / num_runs
        throughput = batch_size / avg_time
        
        print(f"Batch size {batch_size}: {avg_time:.3f}s per batch, {throughput:.1f} samples/s")
    
    print("✓ Performance benchmark completed!\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("SGWCN Model Test Suite")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        test_basic_functionality()
        test_model_statistics()
        test_graph_builder()
        test_wavelet_conv()
        test_bipolar_lif()
        test_end_to_end()
        benchmark_performance()
        
        print("=" * 60)
        print("🎉 All tests passed successfully!")
        print("SGWCN implementation is ready for training.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 