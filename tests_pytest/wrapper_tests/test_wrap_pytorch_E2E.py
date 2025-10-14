"""
End-to-End Testing for MCTWrapper with PyTorch Framework

This module provides comprehensive end-to-end tests for the MCTWrapper
quantization functionality using PyTorch models. It tests various
quantization methods including PTQ, GPTQ, and their mixed-precision variants.

Test Coverage:
- Post-Training Quantization (PTQ)
- PTQ with Mixed Precision (MixP)
- Gradient Post-Training Quantization (GPTQ)
- GPTQ with Mixed Precision (MixP)

The tests use MobileNetV2 as the target model and ImageNet validation
dataset for representative data and accuracy evaluation. All quantized
models are exported to ONNX format for cross-platform deployment.
"""

# Import required libraries
import pytest
import os
import torch
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.datasets import ImageNet
from tqdm import tqdm
from typing import Callable, Generator, List, Tuple, Any

# Import MCT core
import model_compression_toolkit as mct
from model_compression_toolkit.core import QuantizationErrorMethod


@pytest.fixture(scope="session")
def imagenet_dataset() -> Callable[[int, bool], DataLoader]:
    """
    Setup ImageNet dataset for PyTorch testing.
    
    This fixture handles ImageNet dataset preparation including download,
    extraction, and PyTorch DataLoader creation. It provides a factory
    function to create PyTorch DataLoaders with configurable batch size
    and shuffle options.
    
    Returns:
        function: Factory function to create DataLoader with
                 (batch_size, shuffle) parameters
    """
    # Download ImageNet dataset if not present
    if not os.path.isdir('imagenet'):
        os.system('mkdir imagenet')
        os.system('wget -P imagenet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz')
        os.system('wget -P imagenet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar')

    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    dataset = ImageNet(root='./imagenet', split='val',
                       transform=weights.transforms())
    
    def get_dataloader(batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return get_dataloader


@pytest.fixture
def float_model() -> torch.nn.Module:
    """
    Create a pre-trained MobileNetV2 model for PyTorch quantization testing.
    
    This fixture provides a PyTorch MobileNetV2 model with ImageNet pre-trained
    weights. The model is ready for quantization testing and uses the same
    architecture and weights as the reference implementation.
    
    Returns:
        torch.nn.Module: Pre-trained MobileNetV2 model with ImageNet weights
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    return mobilenet_v2(weights=weights)


@pytest.fixture
def representative_dataset_gen(imagenet_dataset: Callable[[int, bool], DataLoader]) -> Callable[[], Generator[List[torch.Tensor], None, None]]:
    """
    Create representative dataset generator for PyTorch quantization.
    
    This fixture provides a generator function that yields batches of
    representative data for quantization calibration. The data is sampled
    from the ImageNet validation set and prepared in PyTorch tensor format.
    
    The generator is used by MCT to determine optimal quantization parameters
    by analyzing the distribution of activations during forward passes.
    
    Args:
        imagenet_dataset: Fixture providing ImageNet DataLoader factory
        
    Returns:
        function: Generator function that yields [batch_tensor] lists
    """
    batch_size = 10
    n_iter = 5
    dataloader = imagenet_dataset(batch_size, shuffle=True)
    
    def gen() -> Generator[List[torch.Tensor], None, None]:
        dataloader_iter = iter(dataloader)
        for _ in range(n_iter):
            yield [next(dataloader_iter)[0]]
    
    return gen


def evaluate_model(model: torch.nn.Module, testloader: DataLoader) -> float:
    """
    Evaluate PyTorch model accuracy using a DataLoader.
    
    This function performs comprehensive accuracy evaluation of PyTorch models
    on the provided test dataset. It automatically detects and uses GPU if
    available, otherwise falls back to CPU computation.
    
    Args:
        model (torch.nn.Module): PyTorch model to evaluate
        testloader (DataLoader): PyTorch DataLoader with test data
        
    Returns:
        float: Top-1 accuracy percentage (0-100)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = (100 * correct / total)
    print(f'Accuracy: {val_acc:.2f}%')
    return val_acc


@pytest.mark.parametrize("quant_func", [
    "PTQ_Pytorch",
    "PTQ_Pytorch_MixP",
    "GPTQ_Pytorch",
    "GPTQ_Pytorch_MixP",
])
def test_quantization(quant_func: str, imagenet_dataset: Callable[[int, bool], DataLoader], float_model: torch.nn.Module,
                      representative_dataset_gen: Callable[[], Generator[List[torch.Tensor], None, None]]) -> None:
    """
    Test end-to-end quantization workflows for PyTorch models.
    
    This comprehensive test function validates different PyTorch quantization
    methods by executing the complete workflow from model preparation through
    quantization to accuracy evaluation and ONNX export.
 
    Args:
        quant_func (str): Name of quantization method to test
        imagenet_dataset: Fixture providing ImageNet DataLoader factory
        float_model: Fixture providing pre-trained MobileNetV2 model
        representative_dataset_gen: Fixture providing calibration data
                                    generator
        
    Test Methods:
        - PTQ_Pytorch: Standard Post-Training Quantization for PyTorch
        - PTQ_Pytorch_MixP: PTQ with Mixed Precision optimization
        - GPTQ_Pytorch: Gradient-based Post-Training Quantization
        - GPTQ_Pytorch_MixP: GPTQ with Mixed Precision optimization
        
    Export Format:
        All quantized models are exported to ONNX format for cross-platform
        deployment and inference optimization.
    """
    
    # Decorator to print logs before and after function execution
    def decorator(func: Callable[[torch.nn.Module], Tuple[bool, torch.nn.Module]]) -> Callable[[torch.nn.Module], Tuple[bool, torch.nn.Module]]:
        """
        Decorator for logging quantization function execution.
        
        This decorator wraps quantization functions to provide clear logging
        of execution progress and proper error handling. It tracks when each
        quantization method starts and ends, and converts failures into
        clear runtime errors.
        
        Args:
            func: Quantization function to wrap
            
        Returns:
            function: Wrapped function with logging and error handling
        """
        def wrapper(*args: Any, **kwargs: Any) -> Tuple[bool, torch.nn.Module]:
            print(f"----------------- {func.__name__} Start ---------------")
            flag, result = func(*args, **kwargs)
            print(f"----------------- {func.__name__} End -----------------")
            if not flag:
                raise RuntimeError(f"Quantization failed for {func.__name__}")
            return flag, result
        return wrapper

    #########################################################################
    # Run PTQ (Post-Training Quantization) with PyTorch
    @decorator
    def PTQ_Pytorch(float_model):
        """
        Execute Post-Training Quantization (PTQ) on PyTorch model.
        
        PTQ is a quantization method that converts a pre-trained floating-point
        PyTorch model to a quantized model without requiring additional
        training.
        It uses representative data to determine optimal quantization
        parameters
        and exports the result to ONNX format.
        
        Args:
            float_model: Pre-trained floating-point PyTorch model
            
        Returns:
            tuple: (success_flag, quantized_model)
        """
        method = 'PTQ'
        framework = 'pytorch'
        use_MCT_TPC = False
        use_MixP = False

        param_items = [
            ['target_platform_version', 'v1',
             'Target platform capabilities version.'],
            ['activation_error_method', QuantizationErrorMethod.MSE,
             'ErrorMethod.'],
            ['weights_bias_correction', True, ''],
            ['z_threshold', float('inf'), ''],
            ['linear_collapsing', True, ''],
            ['residual_collapsing', True, ''],
            ['save_model_path', './qmodel_PTQ_Pytorch.onnx',
             'Path to save the model.']
        ]

        wrapper = mct.wrapper.mctwrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(
            float_model, method, framework, use_MCT_TPC, use_MixP,
            representative_dataset_gen, param_items)
        return flag, quantized_model

    #########################################################################
    # Run PTQ + Mixed Precision Quantization (MixP) with PyTorch
    @decorator
    def PTQ_Pytorch_MixP(float_model):
        """
        Execute PTQ with Mixed Precision Quantization on PyTorch model.
        
        This method combines Post-Training Quantization with Mixed Precision
        optimization to achieve better accuracy-efficiency trade-offs. It
        automatically determines optimal bit-width allocation for different
        layers based on their sensitivity to quantization.
        
        Args:
            float_model: Pre-trained floating-point PyTorch model
            
        Returns:
            tuple: (success_flag, quantized_model)
        """
        method = 'PTQ'
        framework = 'pytorch'
        use_MCT_TPC = False
        use_MixP = True

        param_items = [
            ['target_platform_version', 'v1',
             'Target platform capabilities version.'],
            ['num_of_images', 5, 'Number of images for mixed precision.'],
            ['use_hessian_based_scores', False, 'Use Hessian-based scores.'],
            ['weights_compression_ratio', 0.5, 'Compression ratio.'],
            ['save_model_path', './qmodel_PTQ_Pytorch_MixP.onnx',
             'Path to save the model.']
        ]

        wrapper = mct.wrapper.mctwrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(
            float_model, method, framework, use_MCT_TPC, use_MixP,
            representative_dataset_gen, param_items)
        return flag, quantized_model

    #########################################################################
    # Run GPTQ (Gradient-based PTQ) with PyTorch
    @decorator
    def GPTQ_Pytorch(float_model):
        """
        Execute Gradient-based Post-Training Quantization (GPTQ) on
        PyTorch model.
        
        GPTQ is an advanced quantization method that uses gradient-based
        optimization to fine-tune quantization parameters. It iteratively
        adjusts the quantized weights to minimize the loss function, resulting
        in better accuracy preservation compared to standard PTQ.
 
        Args:
            float_model: Pre-trained floating-point PyTorch model
            
        Returns:
            tuple: (success_flag, quantized_model)
        """
        method = 'GPTQ'
        framework = 'pytorch'
        use_MCT_TPC = False
        use_MixP = False

        param_items = [
            ['target_platform_version', 'v1',
             'Target platform capabilities version.'],
            ['n_epochs', 5, 'Number of epochs for fine-tuning.'],
            ['optimizer', None, 'Optimizer for fine-tuning.'],
            ['save_model_path', './qmodel_GPTQ_Pytorch.onnx',
             'Path to save the model.']
        ]

        wrapper = mct.wrapper.mctwrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(
            float_model, method, framework, use_MCT_TPC, use_MixP,
            representative_dataset_gen, param_items)
        return flag, quantized_model

    #########################################################################
    # Run GPTQ + Mixed Precision Quantization (MixP) with PyTorch
    @decorator
    def GPTQ_Pytorch_MixP(float_model):
        """
        Execute GPTQ with Mixed Precision Quantization on PyTorch model.
        
        This method combines Gradient-based Post-Training Quantization with
        Mixed Precision optimization to achieve the best possible accuracy-
        efficiency trade-off. It uses gradient-based fine-tuning while
        automatically selecting optimal bit-widths for different layers.

        Args:
            float_model: Pre-trained floating-point PyTorch model
            
        Returns:
            tuple: (success_flag, quantized_model)
        """
        method = 'GPTQ'
        framework = 'pytorch'
        use_MCT_TPC = False
        use_MixP = True

        param_items = [
            ['target_platform_version', 'v1',
             'Target platform capabilities version.'],
            ['n_epochs', 5, 'Number of epochs for fine-tuning.'],
            ['optimizer', None, 'Optimizer for fine-tuning.'],
            ['num_of_images', 5, 'Number of images for mixed precision.'],
            ['use_hessian_based_scores', False, 'Use Hessian-based scores.'],
            ['weights_compression_ratio', 0.5, 'Compression ratio.'],
            ['save_model_path', './qmodel_GPTQ_Pytorch_MixP.onnx',
             'Path to save the model.']
        ]

        wrapper = mct.wrapper.mctwrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(
            float_model, method, framework, use_MCT_TPC, use_MixP,
            representative_dataset_gen, param_items)
        return flag, quantized_model

    # Execute the selected quantization method
    quant_methods = {
        "PTQ_Pytorch": PTQ_Pytorch,
        "PTQ_Pytorch_MixP": PTQ_Pytorch_MixP,
        "GPTQ_Pytorch": GPTQ_Pytorch,
        "GPTQ_Pytorch_MixP": GPTQ_Pytorch_MixP,
    }
    
    # Run the quantization method and verify success
    flag, quantized_model = quant_methods[quant_func](float_model)
    assert flag, f"Quantization failed for {quant_func}"

    # Validation: Evaluate quantized model accuracy on ImageNet
    val_dataloader = imagenet_dataset(50, shuffle=False)
    quantized_accuracy = evaluate_model(quantized_model, val_dataloader)
    
    # Display results with detailed accuracy information
    print(f"{quant_func} Quantized model's Top 1 accuracy on the ImageNet "
          f"validation set: {quantized_accuracy:.2f}%")

    # Assert minimum accuracy threshold to ensure quantization quality
    assert quantized_accuracy >= 30.0, \
        f"Accuracy too low: {quantized_accuracy:.2f}%"


if __name__ == '__main__':
    """Run tests when script is executed directly."""
    pytest.main([__file__])
