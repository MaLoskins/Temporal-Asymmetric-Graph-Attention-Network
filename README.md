# Temporal Asymmetric Geometric Attention Network (TAGAN)

## Overview

TAGAN is a novel deep learning architecture that combines geometric attention mechanisms with asymmetric temporal processing for analyzing dynamic graph structures over time. This implementation provides a comprehensive Python framework for training, evaluating, and visualizing temporal graph neural networks with asymmetric dependencies.

## Key Features

- **Temporal Encoding**: Custom mechanisms to capture asymmetric dependencies in temporal data
- **Geometric Attention**: Multi-head attention modules with various distance metrics
- **Memory Bank**: Efficient tracking of node states across temporal snapshots
- **Encoder-Decoder Structure**: Flexible architecture with skip connections
- **Visualization Tools**: Rich visualization capabilities for attention patterns

## Architecture

The TAGAN architecture consists of several key components:

1. **Node Memory Bank**: Stores and updates node embeddings across time
2. **Geometric Attention Layer**: Processes spatial relationships within each time step
3. **Temporal Propagation Layer**: Propagates information asymmetrically through time
4. **Temporal Attention**: Aggregates information across time steps
5. **Classification/Regression Layer**: Produces final predictions

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- NumPy
- Matplotlib
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage Example

```python
from src.tagan.model import TAGAN
from src.tagan.utils.config import TAGANConfig
from src.tagan.training.trainer import TAGANTrainer

# Configure model
config = TAGANConfig(
    node_feature_dim=16,
    edge_feature_dim=8,
    hidden_dim=64,
    output_dim=1,
    num_heads=4
)

# Create model
model = TAGAN(config)

# Train model
trainer = TAGANTrainer(model, config)
trainer.train(train_loader, val_loader)

# Evaluate model
results = trainer.test(test_loader)

# Visualize attention
attention_output = model.infer_with_attention(sample_data)
```

See `example.py` for a complete working example.

## Module Structure

- `src/tagan/model.py`: Main TAGAN model implementation
- `src/tagan/layers/`: Neural network layers
  - `geometric_attention.py`: Spatial attention mechanisms
  - `temporal_attention.py`: Temporal attention mechanisms
  - `temporal_propagation.py`: Temporal propagation layers
  - `classification.py`: Output layers and loss functions
- `src/tagan/utils/`: Utility functions
  - `memory_bank.py`: Node state tracking
  - `config.py`: Configuration management
  - `metrics.py`: Evaluation metrics
- `src/tagan/data/`: Data processing
  - `data_loader.py`: Data loading utilities
  - `preprocessing.py`: Data preprocessing functions
- `src/tagan/visualization/`: Visualization tools
  - `attention_vis.py`: Attention pattern visualization
- `src/tagan/training/`: Training utilities
  - `trainer.py`: Model training and evaluation

## Extending TAGAN

The modular design allows for easy extension:

1. **Custom Distance Metrics**: Add new distance functions to `DistanceMetric` class
2. **Temporal Encodings**: Implement new encoding schemes in `TimeEncoding` class
3. **Attention Mechanisms**: Create new attention modules that inherit from base classes
4. **Loss Functions**: Add custom loss functions to `TAGANLoss` class

## Citation

If you use TAGAN in your research, please cite:

```
@article{tagan2025,
  title={Temporal Asymmetric Geometric Attention Networks for Dynamic Graph Analysis},
  author={TAGAN Team},
  journal={Journal of Machine Learning Research},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.