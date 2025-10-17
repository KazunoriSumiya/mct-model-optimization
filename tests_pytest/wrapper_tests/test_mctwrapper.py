"""
Test cases for MCTWrapper class from model_compression_toolkit.wrapper.mctwrapper
"""

import pytest
from unittest.mock import Mock, patch
from typing import Any, List, Tuple

# Add sys.path to allow local package import
import sys
print(sys.path)
sys.path.append('/home/ubuntu/wrapper/sonyfork/mct-model-optimization')
print(sys.path)

from model_compression_toolkit.core import QuantizationErrorMethod
from model_compression_toolkit.wrapper.mctwrapper import MCTWrapper


class TestMCTWrapper:
    """
    Unit Tests for MCTWrapper Core Functionality
    
    This test class focuses on testing individual methods and components
    of the MCTWrapper class in isolation. Each test uses mocking to avoid
    dependencies on external libraries and focuses on specific functionality.
    
    Test Categories:
        - Input Validation: Testing _initialize_and_validate success cases
        - Parameter Management: Testing _modify_params functionality
        - TPC Configuration: Testing _get_TPC with different TPC sources
        - Method Selection: Testing _select_method for different frameworks
        - Configuration Methods: Testing PTQ/GPTQ parameter generation
        - Export Functionality: Testing model export for different frameworks
    """

    def test_initialize_and_validate_valid_inputs(self) -> None:
        """
        Test _initialize_and_validate method with valid input parameters.
        
        This test verifies that the _initialize_and_validate method correctly
        initializes all wrapper instance attributes when provided with valid
        input parameters for all supported configurations.
        """
        wrapper = MCTWrapper()
        mock_model = Mock()
        mock_dataset = Mock()
        
        wrapper._initialize_and_validate(
            mock_model, 'PTQ', 'pytorch', True, False, mock_dataset)
        
        assert wrapper.float_model == mock_model
        assert wrapper.method == 'PTQ'
        assert wrapper.framework == 'pytorch'
        assert wrapper.use_MCT_TPC is True
        assert wrapper.use_MixP is False
        assert wrapper.representative_dataset == mock_dataset

    def test_modify_params(self) -> None:
        """
        Test _modify_params method with existing parameter keys.
        
        This test verifies that the _modify_params method correctly updates
        existing parameters in the wrapper's params dictionary when given
        valid parameter items.
        """
        wrapper = MCTWrapper()
        
        # Prepare test parameter items with existing keys
        param_items = [
            ('n_epochs', 10, 'Number of epochs'),
            ('learning_rate', 0.01, 'Learning rate'),
            ('fw_name', 'tensorflow', 'Framework name')
        ]
        
        # Call _modify_params to update existing parameters
        wrapper._modify_params(param_items)
        
        # Verify that parameters were updated correctly
        assert wrapper.params['n_epochs'] == 10
        assert wrapper.params['learning_rate'] == 0.01
        assert wrapper.params['fw_name'] == 'tensorflow'

    def test_modify_params_non_existing_keys(self) -> None:
        """
        Test _modify_params method with non-existing parameter keys.
        
        This test verifies that the _modify_params method correctly handles
        parameter items that contain keys not present in the wrapper's default
        params dictionary. The method should ignore unknown keys and preserve
        all existing parameters unchanged.
        """
        wrapper = MCTWrapper()
        original_params = wrapper.params.copy()
        
        # Prepare test parameter items with non-existing keys
        param_items = [
            ('non_existing_key', 'value', 'Description'),
            ('another_fake_key', 42, 'Another description')
        ]
        
        # Call _modify_params
        wrapper._modify_params(param_items)
        
        # Check that original parameters are unchanged
        assert wrapper.params == original_params
        assert 'non_existing_key' not in wrapper.params
        assert 'another_fake_key' not in wrapper.params

    @patch('model_compression_toolkit.wrapper.mctwrapper.mct.get_target_platform_capabilities')
    def test_get_TPC_with_MCT_TPC(self, mock_mct_get_tpc: Mock) -> None:
        """
        Test _get_TPC method when using MCT TPC.
        
        Verifies that when use_MCT_TPC is True, the wrapper correctly calls
        mct.get_target_platform_capabilities with expected parameters.
        
        Note: Patch targets mct.get_target_platform_capabilities because
        MCTWrapper imports 'model_compression_toolkit as mct'.
        """
        wrapper = MCTWrapper()
        wrapper.use_MCT_TPC = True
        mock_tpc = Mock()
        mock_mct_get_tpc.return_value = mock_tpc
        
        wrapper._get_TPC()
        
        # Check if MCT get_target_platform_capabilities was called correctly
        # These parameters match the default values in MCTWrapper.__init__()
        expected_params = {
            'fw_name': 'pytorch',
            'target_platform_name': 'imx500',
            'target_platform_version': 'v1'
        }
        mock_mct_get_tpc.assert_called_once_with(**expected_params)
        assert wrapper.tpc == mock_tpc

    @patch('model_compression_toolkit.wrapper.mctwrapper.'
           'edgemdt_tpc.get_target_platform_capabilities')
    def test_get_TPC_without_MCT_TPC(self, mock_edgemdt_get_tpc: Mock) -> None:
        """
        Test _get_TPC method when using EdgeMDT TPC.
        
        Verifies that when use_MCT_TPC is False, the wrapper correctly calls
        edgemdt_tpc.get_target_platform_capabilities with expected parameters.
        
        Note: Patch targets edgemdt_tpc.get_target_platform_capabilities
        because MCTWrapper imports edgemdt_tpc directly.
        """
        wrapper = MCTWrapper()
        wrapper.use_MCT_TPC = False
        mock_tpc = Mock()
        mock_edgemdt_get_tpc.return_value = mock_tpc
        
        wrapper._get_TPC()
        
        # Check if edgemdt_tpc get_target_platform_capabilities was called
        # These parameters match the default values in MCTWrapper.__init__()
        expected_params = {
            'tpc_version': '1.0',
            'device_type': 'imx500',
            'extended_version': None
        }
        mock_edgemdt_get_tpc.assert_called_once_with(**expected_params)
        assert wrapper.tpc == mock_tpc

    @patch('model_compression_toolkit.core.keras_resource_utilization_data')
    @patch('model_compression_toolkit.ptq.keras_post_training_quantization')
    @patch('model_compression_toolkit.gptq.'
           'keras_gradient_post_training_quantization')
    @patch('model_compression_toolkit.gptq.get_keras_gptq_config')
    @patch('model_compression_toolkit.exporter.keras_export_model')
    def test_select_method_tensorflow_PTQ(
            self, mock_keras_export: Mock, mock_keras_gptq_config: Mock,
            mock_keras_gptq: Mock, mock_keras_ptq: Mock, mock_keras_ru_data: Mock) -> None:
        """
        Test _select_method method for TensorFlow framework with PTQ method.
        
        This test verifies that the _select_method method correctly configures
        all framework-specific function assignments when the wrapper is set to
        use TensorFlow (Keras) framework with Post-Training Quantization (PTQ).
        """
        wrapper = MCTWrapper()
        wrapper.framework = 'tensorflow'
        wrapper.method = 'PTQ'
        wrapper.use_MixP = False
        
        wrapper._select_method()
        
        assert wrapper.params['fw_name'] == 'tensorflow'
        assert wrapper.resource_utilization_data == mock_keras_ru_data
        assert wrapper._post_training_quantization == mock_keras_ptq
        assert wrapper.get_gptq_config == mock_keras_gptq_config
        assert wrapper.export_model == mock_keras_export

    @patch('model_compression_toolkit.core.pytorch_resource_utilization_data')
    @patch('model_compression_toolkit.ptq.pytorch_post_training_quantization')
    @patch('model_compression_toolkit.gptq.'
           'pytorch_gradient_post_training_quantization')
    @patch('model_compression_toolkit.gptq.get_pytorch_gptq_config')
    @patch('model_compression_toolkit.exporter.pytorch_export_model')
    def test_select_method_pytorch_GPTQ(
            self, mock_pytorch_export: Mock, mock_pytorch_gptq_config: Mock,
            mock_pytorch_gptq: Mock, mock_pytorch_ptq: Mock, mock_pytorch_ru_data: Mock) -> None:
        """
        Test _select_method method for PyTorch framework with GPTQ method.
        
        This test verifies that the _select_method method correctly configures
        all framework-specific function assignments when the wrapper is set to
        use PyTorch framework with Gradient Post-Training Quantization (GPTQ).
        """
        wrapper = MCTWrapper()
        wrapper.framework = 'pytorch'
        wrapper.method = 'GPTQ'
        wrapper.use_MixP = False
        
        wrapper._select_method()
        
        assert wrapper.params['fw_name'] == 'pytorch'
        assert wrapper.resource_utilization_data == mock_pytorch_ru_data
        assert wrapper._post_training_quantization == mock_pytorch_gptq
        assert wrapper.get_gptq_config == mock_pytorch_gptq_config
        assert wrapper.export_model == mock_pytorch_export

    @patch('model_compression_toolkit.core.MixedPrecisionQuantizationConfig')
    @patch('model_compression_toolkit.core.CoreConfig')
    @patch('model_compression_toolkit.core.ResourceUtilization')
    def test_setting_PTQ_MixP(
            self, mock_resource_util: Mock, mock_core_config: Mock,
            mock_mixed_precision_config: Mock) -> None:
        """
        Test _Setting_PTQ_MixP method for Mixed Precision PTQ configuration.
        
        This test verifies that the _Setting_PTQ_MixP method correctly configures
        mixed precision Post-Training Quantization parameters by properly setting
        up configuration objects and resource utilization constraints.
        """
        wrapper = MCTWrapper()
        wrapper.float_model = Mock()
        wrapper.representative_dataset = Mock()
        wrapper.tpc = Mock()
        wrapper.framework = 'tensorflow'
        
        # Mock resource utilization data
        mock_ru_data = Mock()
        mock_ru_data.weights_memory = 1000
        wrapper.resource_utilization_data = Mock(return_value=mock_ru_data)
        
        # Mock config objects
        mock_mp_config_instance = Mock()
        mock_mixed_precision_config.return_value = mock_mp_config_instance
        mock_ptq_config_instance = Mock()
        mock_core_config.return_value = mock_ptq_config_instance
        mock_resource_util_instance = Mock()
        mock_resource_util.return_value = mock_resource_util_instance
        
        result = wrapper._setting_PTQ_MixP()
        
        # Verify the method calls
        mock_mixed_precision_config.assert_called_with(
            num_of_images=5,
            use_hessian_based_scores=False
        )
        mock_core_config.assert_called_with(
            mixed_precision_config=mock_mp_config_instance)
        mock_resource_util.assert_called_with(750.0)  # 1000 * 0.75
        
        # Check result structure
        assert 'in_model' in result
        assert 'representative_data_gen' in result
        assert 'target_resource_utilization' in result
        assert 'core_config' in result
        assert 'target_platform_capabilities' in result

    @patch('model_compression_toolkit.core.QuantizationConfig')
    @patch('model_compression_toolkit.core.CoreConfig')
    def test_setting_PTQ(self, mock_core_config: Mock, mock_quant_config: Mock) -> None:
        """
        Test _Setting_PTQ method for standard Post-Training Quantization.
        
        This test verifies that the _Setting_PTQ method correctly configures
        standard Post-Training Quantization parameters without mixed precision,
        focusing on fixed-precision quantization with comprehensive error
        minimization and optimization techniques.
        """
        wrapper = MCTWrapper()
        wrapper.float_model = Mock()
        wrapper.representative_dataset = Mock()
        wrapper.tpc = Mock()
        wrapper.framework = 'pytorch'
        
        # Mock config objects
        mock_quant_config_instance = Mock()
        mock_quant_config.return_value = mock_quant_config_instance
        mock_ptq_config_instance = Mock()
        mock_core_config.return_value = mock_ptq_config_instance
        
        result = wrapper._setting_PTQ()
        
        # Verify the method calls
        mock_quant_config.assert_called_with(
            activation_error_method=QuantizationErrorMethod.MSE,
            weights_error_method=QuantizationErrorMethod.MSE,
            weights_bias_correction=True,
            z_threshold=float('inf'),
            linear_collapsing=True,
            residual_collapsing=True
        )
        mock_core_config.assert_called_with(
            quantization_config=mock_quant_config_instance)
        
        # Check result structure for PyTorch (in_module instead of in_model)
        assert 'in_module' in result
        assert 'in_model' not in result
        assert result['target_resource_utilization'] is None

    def test_setting_GPTQ_pytorch_framework(self) -> None:
        """
        Test _Setting_GPTQ method for PyTorch framework configuration.
        
        This test verifies that the _Setting_GPTQ method correctly configures
        Gradient Post-Training Quantization (GPTQ) parameters specifically for
        PyTorch framework, ensuring proper parameter mapping and framework-
        specific API compatibility.
        """
        wrapper = MCTWrapper()
        wrapper.float_model = Mock()
        wrapper.representative_dataset = Mock()
        wrapper.tpc = Mock()
        wrapper.framework = 'pytorch'
        wrapper.get_gptq_config = Mock(return_value=Mock())
        
        result = wrapper._setting_GPTQ()
        
        # Check that PyTorch-specific parameter mapping is applied
        assert 'model' in result
        assert 'in_model' not in result
        assert result['model'] == wrapper.float_model

    def test_setting_GPTQ_tensorflow_framework(self) -> None:
        """
        Test _Setting_GPTQ method for TensorFlow framework configuration.
        
        This test verifies that the _Setting_GPTQ method correctly configures
        Gradient Post-Training Quantization (GPTQ) parameters specifically for
        TensorFlow/Keras framework, ensuring proper parameter mapping and
        framework-specific API compatibility.
        """
        wrapper = MCTWrapper()
        wrapper.float_model = Mock()
        wrapper.representative_dataset = Mock()
        wrapper.tpc = Mock()
        wrapper.framework = 'tensorflow'
        wrapper.get_gptq_config = Mock(return_value=Mock())
        
        result = wrapper._setting_GPTQ()
        
        # Check that TensorFlow keeps 'in_model' parameter
        assert 'in_model' in result
        assert 'model' not in result
        assert result['in_model'] == wrapper.float_model

    def test_export_model_tensorflow(self) -> None:
        """
        Test _export_model method for TensorFlow framework export functionality.
        
        This test verifies that the _export_model method correctly exports
        quantized TensorFlow/Keras models to TensorFlow Lite format with
        appropriate parameters and framework-specific configurations.
        """
        wrapper = MCTWrapper()
        wrapper.framework = 'tensorflow'
        wrapper.params['save_model_path'] = './test_model.tflite'
        wrapper.representative_dataset = Mock()
        wrapper.export_model = Mock()
        
        mock_quantized_model = Mock()
        
        wrapper._export_model(mock_quantized_model)
        
        # Verify export function was called with correct parameters
        wrapper.export_model.assert_called_once()
        call_args = wrapper.export_model.call_args[1]  # Get keyword arguments
        assert call_args['model'] == mock_quantized_model
        assert call_args['save_model_path'] == './test_model.tflite'

    def test_export_model_pytorch(self) -> None:
        """
        Test _export_model method for PyTorch framework export functionality.
        
        This test verifies that the _export_model method correctly exports
        quantized PyTorch models to ONNX format with appropriate parameters
        and framework-specific configurations for cross-platform deployment.
        """
        wrapper = MCTWrapper()
        wrapper.framework = 'pytorch'
        wrapper.params['save_model_path'] = './test_model.onnx'
        wrapper.representative_dataset = Mock()
        wrapper.export_model = Mock()
        
        mock_quantized_model = Mock()
        
        wrapper._export_model(mock_quantized_model)
        
        # Verify export function was called with correct parameters
        wrapper.export_model.assert_called_once()
        call_args = wrapper.export_model.call_args[1]  # Get keyword arguments
        assert call_args['model'] == mock_quantized_model
        assert call_args['save_model_path'] == './test_model.onnx'
        assert call_args['repr_dataset'] == wrapper.representative_dataset


class TestMCTWrapperIntegration:
    """
    Integration Tests for MCTWrapper Complete Workflows
    
    This test class focuses on testing the complete quantization and export
    workflows by testing the main quantize_and_export method with different
    configurations and scenarios.
    
    Test Categories:
        - PTQ Workflow: Complete Post-Training Quantization flow
        - GPTQ Mixed Precision: Gradient PTQ with mixed precision
        - LQ-PTQ TensorFlow: Low-bit quantization specific to TensorFlow
    """

    @patch('model_compression_toolkit.wrapper.mctwrapper.'
           'MCTWrapper._get_TPC')
    @patch('model_compression_toolkit.wrapper.mctwrapper.'
           'MCTWrapper._select_method')
    @patch('model_compression_toolkit.wrapper.mctwrapper.'
           'MCTWrapper._setting_PTQ')
    @patch('model_compression_toolkit.wrapper.mctwrapper.'
           'MCTWrapper._export_model')
    def test_quantize_and_export_PTQ_flow(
            self, mock_export: Mock, mock_setting_ptq: Mock,
            mock_select_method: Mock, mock_get_tpc: Mock) -> None:
        """
        Test complete quantize_and_export workflow for PTQ.
        
        This integration test verifies the complete PTQ workflow from input
        validation through model export. It mocks internal methods to focus
        on workflow coordination and method call sequences.
        """
        wrapper = MCTWrapper()
        
        # Setup mocks
        mock_float_model = Mock()
        mock_representative_dataset = Mock()
        mock_quantized_model = Mock()
        mock_info = Mock()
        
        # Mock the post_training_quantization method
        wrapper._post_training_quantization = Mock(
            return_value=(mock_quantized_model, mock_info))
        wrapper.export_model = Mock()
        wrapper._setting_PTQparam = mock_setting_ptq
        
        mock_setting_ptq.return_value = {'mock': 'params'}
        
        param_items = [('n_epochs', 10, 'Test parameter')]
        
        # Call the method
        success, result_model = wrapper.quantize_and_export(
            float_model=mock_float_model,
            method='PTQ',
            framework='pytorch',
            use_MCT_TPC=True,
            use_MixP=False,
            representative_dataset=mock_representative_dataset,
            param_items=param_items
        )
        
        # Verify the flow
        assert wrapper.float_model == mock_float_model
        assert wrapper.framework == 'pytorch'
        assert wrapper.representative_dataset == mock_representative_dataset
        
        mock_get_tpc.assert_called_once_with()
        mock_select_method.assert_called_once_with()
        mock_setting_ptq.assert_called_once()
        wrapper._post_training_quantization.assert_called_once_with(
            **{'mock': 'params'})
        mock_export.assert_called_once_with(mock_quantized_model)
        
        assert success is True
        assert result_model == mock_quantized_model

    @patch('model_compression_toolkit.wrapper.mctwrapper.'
           'MCTWrapper._get_TPC')
    @patch('model_compression_toolkit.wrapper.mctwrapper.'
           'MCTWrapper._select_method')
    @patch('model_compression_toolkit.wrapper.mctwrapper.'
           'MCTWrapper._setting_GPTQ_MixP')
    @patch('model_compression_toolkit.wrapper.mctwrapper.'
           'MCTWrapper._export_model')
    def test_quantize_and_export_GPTQ_MixP_flow(
            self, mock_export: Mock, mock_setting_gptq_mixp: Mock,
            mock_select_method: Mock, mock_get_tpc: Mock) -> None:
        """
        Test complete quantize_and_export workflow for GPTQ with Mixed Precision.
        
        This integration test verifies the complete GPTQ (Gradient Post-Training
        Quantization) workflow with Mixed Precision from input validation through
        model export. It focuses on testing the advanced quantization pipeline
        that combines gradient-based optimization with automatic bit-width selection.
        """
        wrapper = MCTWrapper()
        
        # Setup mocks
        mock_float_model = Mock()
        mock_representative_dataset = Mock()
        mock_quantized_model = Mock()
        mock_info = Mock()
        
        wrapper._post_training_quantization = Mock(
            return_value=(mock_quantized_model, mock_info))
        wrapper.export_model = Mock()
        wrapper._setting_PTQparam = mock_setting_gptq_mixp
        
        mock_setting_gptq_mixp.return_value = {'mock': 'gptq_params'}
        
        # Call the method
        success, result_model = wrapper.quantize_and_export(
            float_model=mock_float_model,
            method='GPTQ',
            framework='tensorflow',
            use_MCT_TPC=False,
            use_MixP=True,
            representative_dataset=mock_representative_dataset,
            param_items=[]
        )
        
        # Verify the flow
        mock_get_tpc.assert_called_once_with()
        mock_select_method.assert_called_once_with()
        mock_setting_gptq_mixp.assert_called_once()
        wrapper._post_training_quantization.assert_called_once_with(
            **{'mock': 'gptq_params'})
        mock_export.assert_called_once_with(mock_quantized_model)
        
        assert success is True
        assert result_model == mock_quantized_model

    @patch('model_compression_toolkit.wrapper.mctwrapper.'
           'MCTWrapper._exec_lq_ptq')
    def test_quantize_and_export_LQPTQ_tensorflow(self, mock_exec_lq_ptq: Mock) -> None:
        """
        Test complete quantize_and_export workflow for LQ-PTQ with TensorFlow.
        
        This integration test verifies the complete LQ-PTQ (Low-bit Quantization
        Post-Training Quantization) workflow specifically for TensorFlow framework.
        It tests the specialized quantization path that bypasses the standard
        workflow and uses a dedicated low-bit quantization execution method.
        """
        wrapper = MCTWrapper()
        
        mock_float_model = Mock()
        mock_representative_dataset = Mock()
        mock_quantized_model = Mock()
        
        mock_exec_lq_ptq.return_value = mock_quantized_model
        
        # Call the method
        success, result_model = wrapper.quantize_and_export(
            float_model=mock_float_model,
            method='LQPTQ',
            framework='tensorflow',
            use_MCT_TPC=True,
            use_MixP=False,
            representative_dataset=mock_representative_dataset,
            param_items=[]
        )
        
        # Verify the flow
        mock_exec_lq_ptq.assert_called_once()
        assert success is True
        assert result_model == mock_quantized_model

class TestMCTWrapperErrorHandling:
    """
    Error Handling Tests for MCTWrapper
    
    This test class focuses on testing error conditions, invalid inputs,
    to ensure robust error handling throughout the MCTWrapper functionality.
    """

    def test_quantize_and_export_unsupported_method(self) -> None:
        """
        Test quantize_and_export method with unsupported quantization method.
        
        This error handling test verifies that the MCTWrapper correctly validates
        quantization method parameters and raises appropriate exceptions when
        provided with unsupported or invalid method names.
        """
        wrapper = MCTWrapper()
        
        with pytest.raises(Exception) as exc_info:
            wrapper.quantize_and_export(
                float_model=Mock(),
                method='UNSUPPORTED_METHOD',
                framework='pytorch',
                use_MCT_TPC=True,
                use_MixP=False,
                representative_dataset=Mock(),
                param_items=[]
            )
        
        expected_msg = "Only PTQ, GPTQ and LQPTQ are supported now"
        assert expected_msg in str(exc_info.value)

    def test_quantize_and_export_LQPTQ_with_pytorch(self) -> None:
        """
        Test quantize_and_export with LQPTQ method and PyTorch framework combination.
        
        This error handling test verifies that MCTWrapper correctly enforces
        framework restrictions for LQ-PTQ (Low-bit Quantization Post-Training
        Quantization) by rejecting PyTorch framework and providing clear error
        messaging about framework compatibility limitations.
        """
        wrapper = MCTWrapper()
        
        with pytest.raises(Exception) as exc_info:
            wrapper.quantize_and_export(
                float_model=Mock(),
                method='LQPTQ',
                framework='pytorch',
                use_MCT_TPC=True,
                use_MixP=False,
                representative_dataset=Mock(),
                param_items=[]
            )
        
        expected_msg = "LQ-PTQ is only supported with tensorflow now"
        assert expected_msg in str(exc_info.value)

    def test_quantize_and_export_unsupported_framework(self) -> None:
        """
        Test quantize_and_export method with unsupported framework parameter.
        
        This error handling test verifies that MCTWrapper correctly validates
        framework parameters and raises appropriate exceptions when provided
        with unsupported or invalid framework names. 
        """
        wrapper = MCTWrapper()
        
        with pytest.raises(Exception) as exc_info:
            wrapper.quantize_and_export(
                float_model=Mock(),
                method='PTQ',
                framework='unsupported',
                use_MCT_TPC=True,
                use_MixP=False,
                representative_dataset=Mock(),
                param_items=[]
            )
        
        expected_msg = "Only tensorflow and pytorch are supported now"
        assert expected_msg in str(exc_info.value)


if __name__ == '__main__':
    pytest.main([__file__])