"""
End-to-End Testing for MCTWrapper with Keras/TensorFlow Framework

This module provides comprehensive end-to-end tests for the MCTWrapper
quantization functionality using Keras/TensorFlow models. It tests various
quantization methods including PTQ, GPTQ, and their mixed-precision variants.

Test Coverage:
- Post-Training Quantization (PTQ)
- PTQ with Mixed Precision (MixP)
- Gradient Post-Training Quantization (GPTQ)
- GPTQ with Mixed Precision (MixP)
- Low-bit Quantization PTQ (LQPTQ)

The tests use MobileNetV2 as the target model and ImageNet validation
dataset for representative data and accuracy evaluation.
"""

import pytest
import os
from pathlib import Path
import tensorflow as tf
import keras
from keras.applications.mobilenet_v2 import MobileNetV2
from typing import Callable, Generator, List, Any, Tuple

# Import setup as needed
import model_compression_toolkit as mct
from model_compression_toolkit.core import QuantizationErrorMethod


@pytest.fixture(scope="session")
def imagenet_dataset() -> Callable[[int, bool], tf.data.Dataset]:
    """
    Setup ImageNet dataset for testing.
    
    This fixture handles ImageNet dataset preparation including download,
    extraction, and preprocessing. It provides a factory function to create
    TensorFlow datasets with configurable batch size and shuffle options.
    
    Returns:
        function: Factory function to create dataset with
                 (batch_size, shuffle) parameters
    """

    # Download and extract ImageNet dataset if not present
    if not os.path.isdir('imagenet'):
        # Create base directory for ImageNet data
        os.system('mkdir imagenet')
        
        # Download ImageNet validation dataset and development kit
        # These files are required for model evaluation and testing
        os.system('wget -P imagenet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz')
        os.system('wget -P imagenet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar')
        
        # Move downloaded files to imagenet directory
        os.system('mv ILSVRC2012_devkit_t12.tar.gz imagenet/')
        os.system('mv ILSVRC2012_img_val.tar imagenet/')

    # Setup ImageNet validation directory structure if not exists
    # This creates the directory structure expected by TensorFlow's image_dataset_from_directory
    if not os.path.isdir('imagenet/val'):
        import subprocess
        # Clone MCT repository temporarily for setup scripts
        subprocess.run(['git', 'clone', 'https://github.com/sony/model_optimization.git', 'temp_mct'])
        # Make ImageNet preparation script executable
        os.system('chmod +x tutorials/resources/scripts/prepare_imagenet.sh')
        # Run the preparation script to organize ImageNet data
        subprocess.run(['tutorials/resources/scripts/prepare_imagenet.sh'])

    def imagenet_preprocess_input(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply MobileNetV2-specific preprocessing to input images."""
        return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels

    def get_dataset(batch_size: int, shuffle: bool) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from ImageNet validation images.
        
        Args:
            batch_size: Number of images per batch
            shuffle: Whether to shuffle the dataset
            
        Returns:
            tf.data.Dataset: Preprocessed ImageNet dataset ready for model evaluation
        """
        # Load images from directory structure with specified parameters
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory='./imagenet/val',
            batch_size=batch_size,
            image_size=[224, 224],
            shuffle=shuffle,
            crop_to_aspect_ratio=True,
            interpolation='bilinear')
        
        # Apply MobileNetV2 preprocessing to normalize pixel values
        dataset = dataset.map(lambda x, y: (imagenet_preprocess_input(x, y)), num_parallel_calls=tf.data.AUTOTUNE)
        
        # Prefetch data for improved performance
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
    return get_dataset

@pytest.fixture
def float_model() -> keras.Model:
    """
    Create a pre-trained MobileNetV2 model for quantization testing.
    
    Returns:
        keras.Model: Pre-trained MobileNetV2 model with ImageNet weights
    """
    return MobileNetV2()


@pytest.mark.parametrize("quant_func", [
    "PTQ_Keras",
    "PTQ_Keras_MixP",
    "GPTQ_Keras",
    "GPTQ_Keras_MixP",
    # "LQPTQ_Keras",  # Add if needed
])
def test_quantization(quant_func: str, imagenet_dataset: Callable[[int, bool], tf.data.Dataset], float_model: keras.Model) -> None:
    """
    Test end-to-end quantization workflows for Keras/TensorFlow models.
    
    Args:
        quant_func (str): Name of quantization function to test
        imagenet_dataset: Fixture providing ImageNet dataset factory
        float_model: Fixture providing pre-trained MobileNetV2 model
        
    Test Methods:
        - PTQ_Keras: Standard Post-Training Quantization
        - PTQ_Keras_MixP: PTQ with Mixed Precision optimization
        - GPTQ_Keras: Gradient-based Post-Training Quantization
        - GPTQ_Keras_MixP: GPTQ with Mixed Precision optimization
    """
    # Configuration for representative dataset generation
    batch_size = 5  # Small batch size for faster testing
    n_iter = 2      # Number of iterations for representative data
    
    # Create dataset instance for representative data generation
    dataset = imagenet_dataset(batch_size, shuffle=True)

    def representative_dataset_gen() -> Generator[List[Any], None, None]:
        """
        Generator function for providing representative data during
        quantization.
        
        This generator yields batches of image data that MCT uses to determine
        optimal quantization parameters. The data is sampled from the ImageNet
        validation set and preprocessed for MobileNetV2.
        
        Yields:
            list: Batch of preprocessed images as numpy arrays
        """
        for _ in range(n_iter):
            yield [dataset.take(1).get_single_element()[0].numpy()]

    # Decorator to print logs before and after function execution
    def decorator(func: Callable[[keras.Model], Tuple[bool, keras.Model]]) -> Callable[[keras.Model], Tuple[bool, keras.Model]]:
        """
        Decorator for logging quantization function execution.
        
        This decorator wraps quantization functions to provide clear logging
        of when each quantization method starts and ends, and handles any
        failures by terminating execution.
        
        Args:
            func: Quantization function to wrap
            
        Returns:
            function: Wrapped function with logging capabilities
        """
        def wrapper(*args: Any, **kwargs: Any) -> Tuple[bool, keras.Model]:
            print(f"----------------- {func.__name__} Start ---------------")
            flag, result = func(*args, **kwargs)
            print(f"----------------- {func.__name__} End -----------------")
            if not flag:
                exit()
            return flag, result
        return wrapper

    #########################################################################
    # Run PTQ (Post-Training Quantization) with Keras
    @decorator
    def PTQ_Keras(float_model: keras.Model) -> Tuple[bool, keras.Model]:
        """
        Execute Post-Training Quantization using MCT Target Platform Capabilities.
        
        This method applies standard PTQ without mixed precision, using MCT's
        predefined target platform capabilities for optimal quantization settings.
        """
        # Quantization method configuration
        method = 'PTQ'
        framework = 'tensorflow'
        use_MCT_TPC = True
        use_MixP = False

        # Configure quantization parameters for optimal model performance
        param_items = [['tpc_version', '1.0', 'The version of the TPC to use.'],

                       ['activation_error_method', QuantizationErrorMethod.MSE, 'ErrorMethod.'],
                       ['weights_bias_correction', True, ''],
                       ['z_threshold', float('inf'), ''],
                       ['linear_collapsing', True, ''],
                       ['residual_collapsing', True, ''],

                       ['save_model_path', './qmodel_PTQ_Keras.tflite', 'Path to save the model.']]

        # Execute quantization using MCTWrapper
        wrapper = mct.wrapper.mct_wrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(float_model, method, framework, use_MCT_TPC, use_MixP, representative_dataset_gen, param_items)
        return flag, quantized_model

    #########################################################################
    # Run PTQ + Mixed Precision Quantization (MixP) with Keras
    @decorator
    def PTQ_Keras_MixP(float_model: keras.Model) -> Tuple[bool, keras.Model]:
        """
        Execute PTQ with Mixed Precision optimization for better accuracy.
        
        Mixed Precision allows different layers to use different bit-widths,
        optimizing the trade-off between model size and accuracy.
        """
        # Quantization method configuration
        method = 'PTQ'
        framework = 'tensorflow'
        use_MCT_TPC = True
        use_MixP = True

        # Configure mixed precision parameters for optimal compression
        param_items = [['tpc_version', '1.0', 'The version of the TPC to use.'],
                       ['num_of_images', 5, 'Whether to use Hessian-based scores for weighted average distance metric computation. This is identical to passing'],
                       ['use_hessian_based_scores', False, ' Whether to use Hessian-based scores for weighted average distance metric computation. This is identical to passing'],

                       ['weights_compression_ratio', 0.75, ''],

                       ['save_model_path', './qmodel_PTQ_Keras_MixP.tflite', 'Path to save the model.']]

        # Execute quantization with mixed precision using MCTWrapper
        wrapper = mct.wrapper.mct_wrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(float_model, method, framework, use_MCT_TPC, use_MixP, representative_dataset_gen, param_items)
        return flag, quantized_model

    #########################################################################
    # Run GPTQ (Gradient-based PTQ) with Keras
    @decorator
    def GPTQ_Keras(float_model: keras.Model) -> Tuple[bool, keras.Model]:
        """
        Execute Gradient-based Post-Training Quantization for enhanced accuracy.
        
        GPTQ uses gradient information to fine-tune quantization parameters,
        resulting in better model accuracy compared to standard PTQ.
        """
        # Quantization method configuration
        method = 'GPTQ'
        framework = 'tensorflow'
        use_MCT_TPC = False
        use_MixP = False

        # Configure GPTQ-specific parameters for gradient-based optimization
        param_items = [['target_platform_version', 'v1', 'Target platform capabilities version.'],

                       ['n_epochs', 5, 'Number of epochs for running the representative dataset for fine-tuning.'],
                       ['optimizer', None, 'optimizer to use for fine-tuning for auxiliary variable.'],
    
                       ['save_model_path', './qmodel_GPTQ_Keras.tflite', 'Path to save the model.']]

        # Execute gradient-based quantization using MCTWrapper
        wrapper = mct.wrapper.mct_wrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(float_model, method, framework, use_MCT_TPC, use_MixP, representative_dataset_gen, param_items)
        return flag, quantized_model

    #########################################################################
    # Run GPTQ + Mixed Precision Quantization (MixP) with Keras
    @decorator
    def GPTQ_Keras_MixP(float_model: keras.Model) -> Tuple[bool, keras.Model]:
        method = 'GPTQ'
        framework = 'tensorflow'
        use_MCT_TPC = False
        use_MixP = True

        param_items = [['target_platform_version', 'v1', 'Target platform capabilities version.'],

                       ['n_epochs', 5, 'Number of epochs for running the representative dataset for fine-tuning.'],
                       ['optimizer', None, 'optimizer to use for fine-tuning for auxiliary variable.'],

                       ['num_of_images', 5, 'Whether to use Hessian-based scores for weighted average distance metric computation. This is identical to passing'],
                       ['use_hessian_based_scores', False, ' Whether to use Hessian-based scores for weighted average distance metric computation. This is identical to passing'],

                       ['weights_compression_ratio', 0.75, ''],

                       ['save_model_path', './qmodel_GPTQ_Keras_MixP.tflite', 'Path to save the model.']]

        wrapper = mct.wrapper.mct_wrapper.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(float_model, method, framework, use_MCT_TPC, use_MixP, representative_dataset_gen, param_items)
        return flag, quantized_model

    #########################################################################
    # Run LQPTQ (Low-bit Quantizer PTQ) with Keras
    @decorator
    def LQPTQ_Keras(float_model: keras.Model) -> Tuple[bool, keras.Model]:
        method = 'LQPTQ'
        framework = 'tensorflow'
        use_MCT_TPC = False
        use_MixP = False

        param_items = [

                       ['learning_rate', 0.0001, ''],
                       ['converter_ver', 'v3.14', ''],

                       ['save_model_path', './qmodel_LQPTQ_Keras.tflite', 'Path to save the model.']]

        # Get the first batch of image data and extract only the image
        # part as a NumPy array
        representative_dataset = dataset.take(1).get_single_element()[0].numpy()
        wrapper = mct.wrapper.wrap.MCTWrapper()
        flag, quantized_model = wrapper.quantize_and_export(float_model, method, framework, use_MCT_TPC, use_MixP, representative_dataset, param_items)
        return flag, quantized_model

    # Execute the selected quantization method
    quant_methods = {
        "PTQ_Keras": PTQ_Keras,
        "PTQ_Keras_MixP": PTQ_Keras_MixP,
        "GPTQ_Keras": GPTQ_Keras,
        "GPTQ_Keras_MixP": GPTQ_Keras_MixP,
        # "LQPTQ_Keras": LQPTQ_Keras,  # Uncomment if needed
    }
    
    # Run the selected quantization method and verify success
    flag, quantized_model = quant_methods[quant_func](float_model)
    assert flag, f"Quantization failed for method: {quant_func}"

    # Validation: Evaluate quantized model accuracy on ImageNet validation set
    val_dataset = imagenet_dataset(50, shuffle=False)
    
    # Compile quantized model for evaluation
    quantized_model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics="accuracy")
    
    # Evaluate model performance on validation data
    quantized_accuracy = quantized_model.evaluate(val_dataset)
    
    # Display quantization results for user verification
    print(f"{quant_func} Quantized model's Top 1 accuracy on the ImageNet "
          f"validation set: {(quantized_accuracy[1] * 100):.2f}%")
    
    # Assert minimum accuracy threshold to ensure quantization quality
    # This prevents severely degraded models from passing tests
    assert quantized_accuracy[1] >= 0.3, \
        f"Accuracy too low for {quant_func}: {quantized_accuracy[1]:.3f}"


if __name__ == '__main__':
    """Run tests when script is executed directly."""
    pytest.main([__file__])
