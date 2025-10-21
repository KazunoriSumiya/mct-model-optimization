#  Copyright 2025 Sony Semiconductor Solutions, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import os
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import model_compression_toolkit as mct
#import low_bit_quantizer_ptq.ptq as lq_ptq

import importlib
FOUND_TPC = importlib.util.find_spec("edgemdt_tpc") is not None
if FOUND_TPC:
    import edgemdt_tpc
FOUND_TPC = False

class MCTWrapper:
    """
    Wrapper class for Model Compression Toolkit (MCT) quantization and export.

    This class provides a unified interface for various neural network
    quantization methods including Post-Training Quantization (PTQ), Gradient
    Post-Training Quantization (GPTQ), and Low-bit Quantization PTQ (LQ-PTQ).
    It supports both TensorFlow and PyTorch frameworks with optional
    mixed-precision quantization.

    The wrapper manages the complete quantization pipeline from model input to
    quantized model export, handling framework-specific configurations and
    Target Platform Capabilities (TPC) setup.

    Attributes:
        params (dict): Configuration parameters for quantization methods
        float_model: The input float precision model
        method (str): Selected quantization method ('PTQ', 'GPTQ', 'LQPTQ')
        framework (str): Target framework ('tensorflow', 'pytorch')
        use_MCT_TPC (bool): Whether to use MCT's built-in TPC
        use_MixP (bool): Whether to use mixed-precision quantization
        representative_dataset: Calibration dataset for quantization
        tpc: Target Platform Capabilities configuration
    """
    def __init__(self):
        """
        Initialize MCTWrapper with default parameters.
        """
        self.params: Dict[str, Any] = {
            # TPC
            'fw_name': 'pytorch',
            'target_platform_version': 'v1',
            'tpc_version': '5.0',

            # QuantizationConfig
            'activation_error_method': mct.core.QuantizationErrorMethod.MSE,
            'weights_bias_correction': True,
            'z_threshold': float('inf'),
            'linear_collapsing': True,
            'residual_collapsing': True,

            # GradientPTQConfig
            'n_epochs': 5,
            'optimizer': None,

            # MixedPrecisionQuantizationConfig
            'num_of_images': 5,
            'use_hessian_based_scores': False,

            # ResourceUtilization
            'weights_compression_ratio': None,

            # low_bit_quantizer_ptq
            'learning_rate': 0.001,
            'converter_ver': 'v3.14',

            # Export
            'save_model_path': './qmodel.onnx',
            
            # Callback function
            'callback': None
        }

    def _initialize_and_validate(self, float_model: Any, method: str,
                                 framework: str, use_MCT_TPC: bool,
                                 use_MixP: bool,
                                 representative_dataset: Any) -> None:
        """
        Validate inputs and Initialize parameters.

        Args:
            float_model: The float model to be quantized.
            method (str): Quantization method ('PTQ', 'GPTQ', 'LQPTQ').
            framework (str): Target framework ('tensorflow', 'pytorch').
            use_MCT_TPC (bool): Whether to use MCT's built-in TPC.
            use_MixP (bool): Whether to use mixed-precision quantization.
            representative_dataset: Representative dataset for calibration.

        Raises:
            Exception: If method or framework is not supported.
        """
        # error check --------------------------
        if method not in ['PTQ', 'GPTQ', 'LQPTQ']:
            raise Exception("Only PTQ, GPTQ and LQPTQ are supported now")
        if method == 'LQPTQ' and framework != 'tensorflow':
            raise Exception("LQ-PTQ is only supported with tensorflow now") 
        if framework not in ['tensorflow', 'pytorch']:
            raise Exception("Only tensorflow and pytorch are supported now")        
      
        # set parameters --------------------------
        self.float_model = float_model
        self.method = method
        self.framework = framework
        self.use_MCT_TPC = use_MCT_TPC
        self.use_MixP = use_MixP
        self.representative_dataset = representative_dataset

    def _modify_params(self, param_items: List[List[Any]]) -> None:
        """
        Update the internal parameter dictionary with values from param_items.

        Args:
            param_items (list): List of tuples (key, value, description).
                If key exists in self.params, updates its value.
                Non-existing keys are ignored with a warning.

        Note:
            Only parameters that exist in the default parameter dictionary
            will be updated. Unknown parameters are silently ignored.
        """
        for key, value, _ in param_items:
            if key in self.params:
                # Update parameter value if key exists in default parameters
                self.params[key] = value
            else:
                print(f"Warning: The key '{key}' is not found in the default "
                      f"parameters and will be ignored.")
       
    def _select_method(self) -> None:
        """
        Select and set appropriate quantization, export, and config methods.

        Configures framework-specific methods based on the backend
        (Keras/PyTorch) and quantization method (PTQ/GPTQ). Also sets up
        method-specific parameter configuration functions.

        Note:
            This method dynamically assigns methods to instance attributes
            based on self.framework and self.method values.
        """
        if self.framework == 'tensorflow':
            # Set TensorFlow/Keras specific methods and parameters
            self.params['fw_name'] = 'tensorflow'
            self.resource_utilization_data = mct.core.keras_resource_utilization_data
            self.get_gptq_config = mct.gptq.get_keras_gptq_config
            self.export_model = mct.exporter.keras_export_model
        elif self.framework == 'pytorch':
            # Set PyTorch specific methods and parameters
            self.params['fw_name'] = 'pytorch'
            self.resource_utilization_data = mct.core.pytorch_resource_utilization_data
            self.get_gptq_config = mct.gptq.get_pytorch_gptq_config
            self.export_model = mct.exporter.pytorch_export_model
        else:
            raise Exception("Only tensorflow and pytorch are supported now")

        if self.method == 'PTQ':
            # Set Post-Training Quantization methods
            if self.framework == 'tensorflow':
                self._post_training_quantization = mct.ptq.keras_post_training_quantization
            elif self.framework == 'pytorch':
                self._post_training_quantization = mct.ptq.pytorch_post_training_quantization

            if self.use_MixP:
                # Use mixed precision PTQ parameter configuration
                self._setting_PTQparam = self._setting_PTQ_MixP
            else:
                # Use standard PTQ parameter configuration
                self._setting_PTQparam = self._setting_PTQ

        elif self.method == 'GPTQ':
            # Set Gradient Post-Training Quantization methods
            if self.framework == 'tensorflow':
                self._post_training_quantization = mct.gptq.keras_gradient_post_training_quantization
            elif self.framework == 'pytorch':
                self._post_training_quantization = mct.gptq.pytorch_gradient_post_training_quantization

            if self.use_MixP:
                # Use mixed precision GPTQ parameter configuration
                self._setting_PTQparam = self._setting_GPTQ_MixP
            else:
                # Use standard GPTQ parameter configuration
                self._setting_PTQparam = self._setting_GPTQ

    def _get_TPC(self) -> None:
        """
        Configure Target Platform Capabilities (TPC) based on selected option.

        Sets up either MCT's built-in TPC or external EdgeMDT TPC configuration
        for the IMX500 target platform.

        Note:
            This method sets self.tpc attribute with the configured TPC object.
        """
        if self.use_MCT_TPC:
            # Use MCT's built-in TPC configuration
            params_TPC = {
                'fw_name': self.params['fw_name'],
                'target_platform_name': 'imx500',
                'target_platform_version': self.params['target_platform_version'],
            }
            # Get TPC from MCT framework
            self.tpc = mct.get_target_platform_capabilities(**params_TPC)
        else:
            if FOUND_TPC:
                # Use external EdgeMDT TPC configuration
                params_TPC = {
                    'tpc_version': self.params['tpc_version'],
                    'device_type': 'imx500',
                    'extended_version': None
                }
                # Get TPC from EdgeMDT framework
                self.tpc = edgemdt_tpc.get_target_platform_capabilities(**params_TPC)
            else:
                raise Exception("EdgeMDT TPC module is not available.")

    def _setting_PTQ_MixP(self) -> Dict[str, Any]:
        """
        Generate parameter dictionary for mixed-precision PTQ.

        Returns:
            dict: Parameter dictionary for PTQ.
        """
        params_MPCfg = {
            'num_of_images': self.params['num_of_images'],
            'use_hessian_based_scores': self.params['use_hessian_based_scores'],
        }
        mixed_precision_config = mct.core.MixedPrecisionQuantizationConfig(**params_MPCfg)
        core_config = mct.core.CoreConfig(mixed_precision_config=mixed_precision_config)
        params_RUDCfg = {
            'in_model': self.float_model,
            'representative_data_gen': self.representative_dataset,
            'core_config': core_config,
            'target_platform_capabilities': self.tpc
        }
        ru_data = self.resource_utilization_data(**params_RUDCfg)
        weights_compression_ratio = (
            0.75 if self.params['weights_compression_ratio'] is None
            else self.params['weights_compression_ratio'])
        resource_utilization = mct.core.ResourceUtilization(
            ru_data.weights_memory * weights_compression_ratio)

        params_PTQ = {
            'in_model': self.float_model,
            'representative_data_gen': self.representative_dataset,
            'target_resource_utilization': resource_utilization,
            'core_config': core_config,
            'target_platform_capabilities': self.tpc
        }
        if self.framework == 'pytorch':
            params_PTQ['in_module'] = params_PTQ['in_model']
            del params_PTQ['in_model']
        return params_PTQ

    def _setting_PTQ(self) -> Dict[str, Any]:
        """
        Generate parameter dictionary for PTQ.

        Returns:
            dict: Parameter dictionary for PTQ.
        """
        params_QCfg = {
            'activation_error_method': self.params['activation_error_method'],
            'weights_error_method': mct.core.QuantizationErrorMethod.MSE,
            'weights_bias_correction': self.params['weights_bias_correction'],
            'z_threshold': self.params['z_threshold'],
            'linear_collapsing': self.params['linear_collapsing'],
            'residual_collapsing': self.params['residual_collapsing']
        }
        q_config = mct.core.QuantizationConfig(**params_QCfg)
        core_config = mct.core.CoreConfig(quantization_config=q_config)
        resource_utilization = None

        params_PTQ = {
            'in_model': self.float_model,
            'representative_data_gen': self.representative_dataset,
            'target_resource_utilization': resource_utilization,
            'core_config': core_config,
            'target_platform_capabilities': self.tpc
        }
        if self.framework == 'pytorch':
            params_PTQ['in_module'] = params_PTQ['in_model']
            del params_PTQ['in_model']
        return params_PTQ

    def _setting_GPTQ_MixP(self) -> Dict[str, Any]:
        """
        Generate parameter dictionary for mixed-precision GPTQ.

        Returns:
            dict: Parameter dictionary for GPTQ.
        """
        params_GPTQCfg = {
            'n_epochs': self.params['n_epochs'],
            'optimizer': self.params['optimizer']
        }
        gptq_config = self.get_gptq_config(**params_GPTQCfg)

        params_MPCfg = {
            'num_of_images': self.params['num_of_images'],
            'use_hessian_based_scores': self.params['use_hessian_based_scores'],
        }
        mixed_precision_config = mct.core.MixedPrecisionQuantizationConfig(**params_MPCfg)
        core_config = mct.core.CoreConfig(mixed_precision_config=mixed_precision_config)
        params_RUDCfg = {
            'in_model': self.float_model,
            'representative_data_gen': self.representative_dataset,
            'core_config': core_config,
            'target_platform_capabilities': self.tpc
        }
        ru_data = self.resource_utilization_data(**params_RUDCfg)
        weights_compression_ratio = (
            0.75 if self.params['weights_compression_ratio'] is None
            else self.params['weights_compression_ratio'])
        resource_utilization = mct.core.ResourceUtilization(
            ru_data.weights_memory * weights_compression_ratio)

        config = mct.core.CoreConfig(
            mixed_precision_config = mixed_precision_config,
            quantization_config = mct.core.QuantizationConfig(concat_threshold_update=True)
        )

        params_GPTQ = {
            'in_model': self.float_model,
            'representative_data_gen': self.representative_dataset,
            'target_resource_utilization': resource_utilization,
            'gptq_config': gptq_config,
            'core_config': config,
            'target_platform_capabilities': self.tpc
        }
        if self.framework == 'pytorch':
            params_GPTQ['model'] = params_GPTQ['in_model']
            del params_GPTQ['in_model']
        return params_GPTQ

    def _setting_GPTQ(self) -> Dict[str, Any]:
        """
        Generate parameter dictionary for GPTQ.

        Returns:
            dict: Parameter dictionary for GPTQ.
        """
        params_GPTQCfg = {
            'n_epochs': self.params['n_epochs'],
            'optimizer': self.params['optimizer']
        }
        gptq_config = self.get_gptq_config(**params_GPTQCfg)

        params_GPTQ = {
            'in_model': self.float_model,
            'representative_data_gen': self.representative_dataset,
            'gptq_config': gptq_config,
            'target_platform_capabilities': self.tpc
        }
        if self.framework == 'pytorch':
            params_GPTQ['model'] = params_GPTQ['in_model']
            del params_GPTQ['in_model']
        return params_GPTQ

    def _exec_lq_ptq(self) -> Any:
        """
        Execute Low-bit Quantization Post-Training Quantization (LQ-PTQ).

        Performs quantization using the low_bit_quantizer_ptq method with
        the configured parameters and representative dataset.

        Returns:
            The quantized model object.

        Note:
            This method requires the lq_ptq module to be imported.
        """
        model_save_dir, output_file_name = os.path.split(
            self.params['save_model_path'])
        
        # Note: lq_ptq module should be imported when using this method
        # q_model = lq_ptq.low_bit_quantizer_ptq(
        #     fp_model=self.float_model,
        #     representative_dataset=self.representative_dataset,
        #     model_save_dir=model_save_dir,
        #     output_file_name=output_file_name,
        #     learning_rate=self.params['learning_rate'],
        #     converter_ver=self.params['converter_ver'],
        #     debug_level='INFO',
        #     debug_detail=False,
        #     overwrite_output_file=True)
        # return q_model
        
        # Placeholder implementation - replace with actual lq_ptq call
        raise NotImplementedError(
            "LQ-PTQ functionality requires lq_ptq module to be imported")

    def _export_model(self, quantized_model: Any) -> None:
        """
        Export the quantized model using appropriate export function.

        Configures export parameters based on the framework and exports
        the quantized model to the specified path.

        Args:
            quantized_model: The quantized model to export.

        Note:
            Export format is framework-specific: TFLite for TensorFlow,
            ONNX for PyTorch.
        """
        if self.framework == 'tensorflow':
            params_Export = {
                'model': quantized_model,
                'save_model_path': self.params['save_model_path'],
                'serialization_format': (mct.exporter.KerasExportSerializationFormat.TFLITE),
                'quantization_format': (mct.exporter.QuantizationFormat.FAKELY_QUANT)
            }
        elif self.framework == 'pytorch':
            params_Export = {
                'model': quantized_model,
                'save_model_path': self.params['save_model_path'],
                'repr_dataset': self.representative_dataset
            }
        self.export_model(**params_Export)

    def quantize_and_export(self, float_model: Any, method: str, framework: str,
                            use_MCT_TPC: bool, use_MixP: bool, representative_dataset: Any,
                            param_items: List[List[Any]]) -> None:
        """
        Main function to perform model quantization and export.

        Args:
            float_model: The float model to be quantized.
            method (str): Quantization method, e.g., 'PTQ' or 'GPTQ' or 'LQ=PTQ
            framework (str): 'tensorflow' or 'pytorch'.
            use_MCT_TPC (bool): Whether to use MCT_TPC.
            use_MixP (bool): Whether to use mixed-precision quantization.
            representative_dataset: Representative dataset for calibration.
            param_items (list): List of parameter settings.

        Returns:
            tuple: (Flag, quantized model)
        """
        try:
            # Step 1: Initialize and validate all input parameters
            self._initialize_and_validate(
                float_model, method, framework, use_MCT_TPC, use_MixP,
                representative_dataset)

            # Step 2: Apply custom parameter modifications
            self._modify_params(param_items)

            # Step 3: Handle LQ-PTQ method separately (TensorFlow only)
            if self.method == 'LQPTQ':
                # Execute Low-bit Quantization Post-Training Quantization
                quantized_model = self._exec_lq_ptq()
                return True, quantized_model

            # Step 4: Select framework-specific quantization methods
            self._select_method()

            # Step 5: Configure Target Platform Capabilities
            self._get_TPC()

            # Step 6: Prepare quantization parameters
            params_PTQ = self._setting_PTQparam()
            
            # Step 7: Execute quantization process (PTQ or GPTQ)
            quantized_model, _ = self._post_training_quantization(**params_PTQ)

            # Step 8: Export quantized model to specified format
            self._export_model(quantized_model)

            # Return success flag and quantized model
            return True, quantized_model

        except Exception as e:
            # Log error details and re-raise the exception to caller
            print(f"Error during quantization and export: {str(e)}")
            raise  # Re-raise the original exception to the caller

        finally:
            pass
