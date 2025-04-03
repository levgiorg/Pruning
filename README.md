# TARDIS-Pruning

TARDIS-Pruning is a lightweight model pruning library built on top of Microsoft NNI for PyTorch models. This component provides an easy-to-use interface for pruning deep neural networks to reduce model size and computational requirements while maintaining performance.

## Technical Description

This component implements neural network pruning techniques to create more efficient models for edge deployment. It uses the L1 norm pruning algorithm as the default method to remove less important weights from the model, resulting in a sparse representation that requires less memory and computational resources.

The pruning implementation:
- Supports various layer types including Linear, Conv2d, Conv3d, and BatchNorm2d
- Allows customization of pruning parameters including sparsity ratio
- Automatically identifies and preserves important layers (such as the output layer)
- Returns a pruned model ready for deployment

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- Microsoft NNI 3.0+
- numpy
- Other required packages specified in requirements.txt

### Hardware Requirements
- CPU or CUDA-compatible GPU (for faster processing)
- Sufficient RAM for working with your target models

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-organization/TARDIS-Pruning.git
   cd TARDIS-Pruning
   ```

2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

3. Add the component to your Python path or install it as a package:
   ```
   pip install -e .
   ```

## Usage

The main function exposed by this library is `prune_model`. Here's how to use it:

```python
from pruning import prune_model
import torch

# Your PyTorch model
model = YourModelClass()

# Pruning parameters
sparse_ratio = 0.5  # 50% of weights will be pruned
input_shape = (1, 3, 224, 224)  # Including batch dimension

# Perform pruning
pruned_model = prune_model(model, sparse_ratio, input_shape)

# Save the pruned model
torch.save(pruned_model, 'pruned_model.pth')

# Optionally export to ONNX for deployment
torch.onnx.export(
    pruned_model,
    torch.randn(input_shape),
    "pruned_model.onnx",
    export_params=True,
    opset_version=11
)
```

### API Documentation

The function to be exposed is imported as follows:

```python
from pruning import prune_model
```

#### Parameters

| Parameter            | Type                | Description                                                                 |
|----------------------|---------------------|-----------------------------------------------------------------------------|
| `model`              | `torch.nn.Module`   | The PyTorch model to be pruned.                                             |
| `sparse_ratio`       | `float`             | The ratio of sparsity to be applied to the model (0.0 to 1.0).              |
| `input_shape`        | `tuple`             | The shape of the input tensor that the model expects (including batch).     |
| `pruned_layer_types` | `list`              | A list of layer types to be considered for pruning (default: `['Linear', 'Conv2d', 'Conv3d', 'BatchNorm2d']`). |
| `exclude_layer_names`| `list` or `None`    | A list of layer names to be excluded from pruning (default: `None`, will automatically detect the last layer). |
| `prunner_choice`     | `object` or `None`  | The choice of pruner to be used (default: `None`, will select `L1NormPruner`).                                |

#### Returns 

| Parameter            | Type                | Description                                                                 |
|----------------------|---------------------|-----------------------------------------------------------------------------|
| `model`              | `torch.nn.Module`   | The pruned PyTorch model                                                    |

### Model Serialization and Loading

When saving and loading pruned models, ensure the model definition is available in the same relative path as when the model was created. For cross-platform compatibility, consider using ONNX format:

```python
# Export to ONNX
model.eval()
dummy_input = torch.randn(input_shape)
torch.onnx.export(model, dummy_input, "model.onnx", export_params=True)
```

## Examples

A demo example is included in `demo.py` which prunes a simple VGG-style model:

```python
import torch
from pruning import prune_model
from models import myVGG

SPARSE_RATIO = 0.5
INPUT_SHAPE = (1, 3, 32, 32)  # Including batch

model = myVGG()
pruned_model = prune_model(model, SPARSE_RATIO, INPUT_SHAPE)
torch.save(pruned_model, 'pruned_model.pth')
```

Run the demo with:
```
python demo.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgment

This work was partially supported by the "Trustworthy And Resilient Decentralised Intelligence For Edge Systems (TaRDIS)" Project, funded by EU HORIZON EUROPE program, under grant agreement No 101093006.

## Contact

For questions, issues, or contributions, please open an issue in the repository's issue tracker.






