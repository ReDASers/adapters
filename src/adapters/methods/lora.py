# Code adapted from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py.
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
from typing import Dict, List, NamedTuple, Optional, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.configuration_utils import PretrainedConfig
from transformers.pytorch_utils import Conv1D

from ..composition import AdapterCompositionBlock, Average, BatchSplit, Parallel, Stack
from ..configuration import LoRAConfig, ModelAdaptersConfig
from .adapter_layer_base import AdapterLayerBase, ComposableAdapterLayerBase
from .modeling import Activation_Function_Class
from .utils import dequantize_bnb_weight


try:
    from bitsandbytes.nn import Int8Params, Linear4bit, Linear8bitLt, Params4bit

    bitsandbytes_available = True
except ImportError:
    bitsandbytes_available = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

'''
****************************************************************
Displaying results for sst-2, K=100...

****************************************************************

ADAPTER CONFIGURATION:

{'alpha': 1, 'beta': 12, 'architecture': 'lora', 'selfattn_lora': True, 'intermediate_lora': True, 'output_lora': True, 'leave_out': [], 'r': 8, 'attn_matrices': ['v', 'k'], 'composition_mode': 'add', 'dropout': 0.1, 'use_gating': False, 'init_weights': -1, 'non_linearity': 'leakyrelu', 'eps': 1e-09, 'rescale_frequency': 3}

****************************************************************

{'eval_loss': [0.8019659519195557, 0.6658837795257568, 0.7392714023590088, 0.9019482731819153, 0.8189060091972351], 'eval_accuracy': [0.8495370370370371, 0.8321759259259259, 0.8136574074074074, 0.8321759259259259, 0.8564814814814815], 'eval_runtime': [1.4017, 1.4257, 1.4427, 1.4885, 1.4305], 'eval_samples_per_second': [622.086, 611.639, 604.441, 585.833, 609.584], 'eval_steps_per_second': [19.975, 19.64, 19.409, 18.811, 19.574], 'epoch': [52.0, 47.0, 46.0, 60.0, 60.0], 'task': ['sst-2', 'sst-2', 'sst-2', 'sst-2', 'sst-2'], 'seed': [0, 42, 123, 1001, 2024], 'k': [100, 100, 100, 100, 100]}

mean_eval_loss: 0.7855950832366944 std_eval_loss: 0.07923681920223546
mean_eval_accuracy: 0.8368055555555556 std_eval_accuracy: 0.015019563145841056
mean_eval_runtime: 1.43782 std_eval_runtime: 0.028626589038863844
mean_eval_samples_per_second: 606.7166 std_eval_samples_per_second: 11.916153449834399
mean_eval_steps_per_second: 19.4818 std_eval_steps_per_second: 0.3826347605746248
mean_epoch: 53.0 std_epoch: 6.06630035524124
mean_seed: 638.0 std_seed: 784.8885271170677

****************************************************************
SUMMARY ROUNDED RESULT FOR GLUE sst-2, K=100: accuracy: 83.7 std_accuracy: 1.50
'''

class LoRA(nn.Module):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
        gating_heads: int = 1,
        location_key: str = None,
    ):
        """
        Initializes the LoRA module.

        Args:
            lora_A_shape (tuple): Shape of the A matrix in LoRA.
            lora_B_shape (tuple): Shape of the B matrix in LoRA.
            config (LoRAConfig): Configuration object for LoRA settings.
            gating_heads (int, optional): Number of gating heads. Defaults to 1.
            location_key (str, optional): Location key for LoRA. Defaults to None.
        """
        super().__init__()
        
        # Ensure the composition mode is 'add'
        assert config.composition_mode == "add", "LoRA module only supports composition_mode='add'."
        # Validate and set the location key
        if self._is_valid_location_key(config, location_key) == False:
            raise ValueError(f"Location key {self.location_key} is not enabled in config or invalid.")
        # Initialize configuration parameters
        self.location_key = location_key 
        self.connections_in = lora_A_shape[-1]
        self.connections_out = lora_B_shape[0]
        self.r = int(config.r)
        
        assert self.r == lora_A_shape[0] == lora_B_shape[1], "r must match the first dimension of A and the second dimension of B."
        # The following is for flexibility; normally, alpha is normally 1 for loria
        self.lora_alpha = float(config.alpha) if config.alpha > 0 else math.sqrt(self.r)
        #  scaling factor is also 1 for loria
        self.scaling = float(self.lora_alpha / self.r) if self.lora_alpha > 1.0 else 1.0
        beta = config.beta if config.beta is not None else int(self.r * 1.5)
        self.bottleneck_size = int(beta * self.r)  
        
        self.composition_mode = config.composition_mode
        self.attn_matrices = config.attn_matrices
        self.use_gating = config.use_gating
        self.non_linearity = config.non_linearity 
        self.sigma = config.sigma
        self.eps = config.eps
        self._delta_w = None  # Placeholder for delta weights

        self.dropout = nn.Dropout(p=config.dropout) if config.dropout > 0.0 else lambda x: x
        
        self.mode: Literal["attention", "dense_fan_out", "dense_fan_in", "noop"] = self._calculation_mode()
        self._layer_specific_setup(lora_A_shape, lora_B_shape)
        # Setup gating mechanism if required
        self._setup_gating_maybe(gating_heads)
        self.batches_per_epoch = self._calculate_batches_per_epoch(config.batch_size, config.training_set_size)
        self.n_batches = 0 # have not trained yet
        

    def _calculate_batches_per_epoch(self, batch_size: Optional[int], training_set_size: Optional[int]) -> int:
        if batch_size is not None and training_set_size is not None:
            batches_per_epoch = training_set_size // batch_size
            if batches_per_epoch < 1:
                logging.warning("Turning off rescaling...")
            return batches_per_epoch
        
        logging.warning("Batch size or training set size is None. \
                        Cannot calculate batches per epoch. Setting to 1. \
                        This may lead to incorrect rescaling and suboptimal performance.")
        return 1
            

    def _is_valid_location_key(self, config, location_key):
        """
        Checks if the given location key is valid based on the config.

        Args:
            config (LoRAConfig): Configuration object for LoRA settings.

        Returns:
            bool: True if the location key is valid, False otherwise.
        """
        if location_key is None:
            logging.warning("Location key must be provided, but is currently None.")
            return False
        if (config.selfattn_lora == False and location_key == "selfattn_lora") or \
           (config.intermediate_lora == False and location_key == "intermediate_lora") or \
           (config.output_lora == False and location_key == "output_lora"):
            logging.warning(f"LoRIA module has location key {location_key} but is not enabled in config.")
            return False
        return True

    def _calculation_mode(self):
        """
        Checks if advanced calculation is possible based on the current configuration.

        Returns:
            stromg: how adapter weights will be handled.
        """
        match self.location_key:
            case "selfattn_lora" if self.connections_in == self.connections_out:
                return "attention"
            case "intermediate_lora" if self.connections_in < self.connections_out:
                return "dense_fan_out"
            case "output_lora" if self.connections_in > self.connections_out:
                return "dense_fan_in"
            case _:
                return "noop"

    
    def _layer_specific_setup(self, lora_A_shape, lora_B_shape):
         # Determine calculation mode and setup accordingly
        match self.mode:
            case "attention":
                self._setup_in_attn(lora_A_shape=lora_A_shape, lora_B_shape=lora_B_shape)
            case "dense_fan_in" | "dense_fan_out":
                self._setup_scaling()
            case _:
                pass

    def _calculate_gain(self, nonlinearity: str):
        match nonlinearity:
            case "leaky_relu" | "leakyrelu" | "prelu":
                return nn.init.calculate_gain("leaky_relu", param=self._get_neg_slope(nonlinearity))
            case "linear" | "snselu"  | "sigmoid":
                return 1.0
            case "tanh":
                return nn.init.calculate_gain("tanh")
            case "selu":
                return nn.init.calculate_gain("selu")
            case "mish":
                return nn.init.calculate_gain("leaky_relu", param=self._get_neg_slope(nonlinearity))
            case "gelu":
                return nn.init.calculate_gain("leaky_relu", param=self._get_neg_slope(nonlinearity))
            case "relu":
                return nn.init.calculate_gain("relu")
            case "relu6" | "elu":
                return math.sqrt(2.0)
            case _:
                return nn.init.calculate_gain("leaky_relu", math.sqrt(5))
            
    def _calculate_std(self, gain, fan):
        return gain / math.sqrt(float(fan))
    
    def _get_sigma_xavier_normal(self, layers, nonlinearity: str = "leakyrelu"):
        sigma = 0.0
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                if i < len(layers) - 1:
                    if not isinstance(layers[i + 1], nn.Linear):
                        gain = self._calculate_gain(nonlinearity)
                    else:
                        gain = 1.0
                else:
                    gain = 1.0
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                sigma = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        return sigma

    def _get_sigma_kaiming_normal(self, 
                                  weights, 
                                  nonlinearity: str = "leakyrelu", 
                                  mode: Literal["fan_in", "fan_out"] = "fan_in"):
        """
        Calculates the sigma value for Kaiming normal initialization.

        Args:
            weights (torch.Tensor): Weights tensor.
            nonlinearity (str, optional): Non-linearity function. Defaults to "gelu".
            mode (Literal["fan_in", "fan_out"], optional): Calculation mode. Defaults to "fan_in".

        Returns:
            float: standard deviation value to which the weights will be initialized under kaiming normal.
        """
        fan = nn.init._calculate_correct_fan(weights, mode)
        gain = self._calculate_gain(nonlinearity)
        std = self._calculate_std(gain, fan)
        return std

            
    def _setup_gating_maybe(self, gating_heads: int):
        """
        Sets up the gating mechanism if use_gating is enabled.

        Args:
            gating_heads (int): Number of gating heads.
        """
        if self.use_gating:
            self.gate = nn.Linear(self.connections_in, gating_heads, dtype=torch.float32)
            std = self._get_sigma_kaiming_normal(self.gate.weight, nonlinearity="sigmoid", mode="fan_in")
            nn.init.normal_(self.gate.weight, std=std)

    def _setup_scaling(self):
        """
        Sets up the basic calculation mode by initializing scaling parameters.
        """
        self.lora_C = nn.Parameter(torch.zeros(self.connections_out, 1, dtype=torch.float32))
        self.scalar_scaler = nn.Parameter(torch.tensor(self.eps, dtype=torch.float32))
        self.sigma = self._estimate_scaling_sigma()
        nn.init.normal_(self.lora_C, mean=1.0, std=self.sigma)

    def _estimate_scaling_sigma(self):
        if self.sigma is None:
            return self._get_sigma_kaiming_normal(self.lora_C, mode="fan_out", nonlinearity=self.non_linearity)
        elif isinstance(self.sigma, str):
            if self.sigma == "loria":
                return math.sqrt(2 / ((1 + (self._get_neg_slope(self.non_linearity)) ** 2) * self.connections_out))
            elif self.sigma == "bert":
                return 0.02
            elif self.sigma == "ia3":
                return 0.0
            else:
                raise ValueError(f"Unknown sigma type: {self.sigma}")
        elif isinstance(self.sigma, float) or isinstance(self.sigma, int):
            return float(self.sigma) if self.sigma > 0 else 0.0
        else:
            raise ValueError(f"Unknown sigma type: {self.sigma}")
       

    def _estimate_attn_sigma(self):
        if self.sigma is None:
            return self._get_sigma_kaiming_normal(self.lora_B, mode="fan_out", nonlinearity=self.non_linearity)
        elif isinstance(self.sigma, str):
            if self.sigma == "loria":
                return self._get_sigma_xavier_normal(self.f, nonlinearity=self.non_linearity)
            elif self.sigma == "bert":
                return 0.02
            elif self.sigma == "ia3":
                return 0.0
            else:
                raise ValueError(f"Unknown sigma type: {self.sigma}")
        elif isinstance(self.sigma, float) or isinstance(self.sigma, int):
            return float(self.sigma) if self.sigma > 0 else 0.0
        else:
            raise ValueError(f"Unknown sigma type: {self.sigma}")

            
    def _setup_in_attn(self, lora_A_shape, lora_B_shape):
        """
        Sets up the full calculation mode by initializing autoencoder and other components.

        Args:
            lora_A_shape (tuple): Shape of the A matrix in LoRA.
            lora_B_shape (tuple): Shape of the B matrix in LoRA.
        """
        self.f = self._get_autoencoder_architecture("NLbLN")
        self._initialize_autoencoder_weights(self.f)
        self._setup_lora_matrices(lora_A_shape=lora_A_shape, lora_B_shape=lora_B_shape)
        self.sigma = self._estimate_attn_sigma()

    def _setup_lora_matrices(self, lora_A_shape, lora_B_shape):
        """
        Sets up the LoRA matrices A and B.

        Args:
            lora_A_shape (tuple): Shape of the A matrix in LoRA.
            lora_B_shape (tuple): Shape of the B matrix in LoRA.
        """
        self.lora_A = nn.Parameter(torch.randn(lora_A_shape))
        self.lora_B = nn.Parameter(torch.zeros(lora_B_shape))
        self._initialize_lora_matrices()

    def _initialize_lora_matrices(self):
        """
        Initializes the LoRA matrices A and B.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _get_neg_slope(self, non_linearity: str = "leakyrelu"):
        """
        Retruns the negative slope for various activation functions.

        Returns:
            float: Negative slope value.
        """
        match non_linearity:
            case "leakyrelu" | "leaky_relu" | "prelu":
                return 1e-2
            case "mish":
                return 3e-4
            case "gelu":
                return 5.1e-4
            case "linear":
                return 1.0
            case _:
                return 0.0

    def _initialize_autoencoder_weights(self, layers: nn.Sequential):
        """
        Initializes the weights of the given layers.
        
        Args:
            layers (nn.Sequential): Sequential model containing the layers.
        """
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                if i < len(layers) - 1:
                    if not isinstance(layers[i + 1], nn.Linear):
                        gain = self._calculate_gain(self.non_linearity)
                    else:
                        gain = 1.0
                else:
                    gain = 1.0
                nn.init.xavier_normal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        

    def _get_autoencoder_architecture(self, arch: str = "NLbLN"):
        """
        Retrieves the autoencoder architecture based on the configuration.

        Returns:
            nn.Sequential: The autoencoder architecture as a sequential model.

        Raises:
            ValueError: If the autoencoder architecture is unknown.
        """
        architectures = {
            "NLbLN": [
                nn.Linear(self.connections_in, self.r),
                Activation_Function_Class(self.non_linearity.lower()),
                nn.Linear(self.r, self.bottleneck_size),
                nn.Linear(self.bottleneck_size, self.r),
                Activation_Function_Class(self.non_linearity.lower()),
                nn.Linear(self.r, self.connections_in),
            ],
        }

        try:
            return nn.Sequential(*architectures[arch])
        except KeyError:
            raise ValueError(f"Unknown autoencoder architecture: {arch}")
        

    @property
    def delta_w(self) -> torch.Tensor:
        """Placeholder for delta_w calculation."""
        return self._delta_w
    
    @delta_w.setter
    def delta_w(self, value: torch.Tensor):
        """Sets the delta_w value."""
        self._delta_w = value

    @delta_w.deleter
    def delta_w(self):
        """Deletes the delta_w value."""
        del self._delta_w

    def _do_rescale(self) -> bool:
        """
        Checks if rescaling is required based on the configuration.

        Returns:
            bool: True if rescaling is required, False otherwise.
        """
        if self.batches_per_epoch < 1:
            return False
        
        self.n_batches += 1

        if self.n_batches > self.batches_per_epoch:
            self.n_batches = 0
            return True
        return False
            
    def _rescale_weights(self):
        """
        Rescale the lora_A and lora_B weights based on the current configuration.
        """
        
        if self.mode in ["dense_fan_in", "dense_fan_out"] and self.batches_per_epoch >= 1:
            self.lora_C.data = self.rescale(self.lora_C.data, sigma=self.sigma, dtype=torch.float32)    
         
    def rescale(self, weights: torch.Tensor, sigma: torch.float32 = 0.05, dtype: torch.dtype = None) -> torch.Tensor:
        if sigma == 0.0:
            return weights
        w = torch.nan_to_num(weights)
        u = torch.mean(w, dtype=dtype)
        stddev = torch.std(w)
        # calculate z-scores
        z = (w - u) / (stddev + 1e-12)
        # rescale to original range
        return z * sigma + u
    
    def com(self, weights: torch.Tensor, added: torch.Tensor, scaling: Optional[float]=None) -> torch.Tensor:
        """Performs the composition operation between existing and injected weights.

        Args:
            weights (torch.Tensor): Existing weights.
            added (torch.Tensor): Weights to add.
            scaling (float, optional): Scaling factor -- left for compatibility with the API. 
                                       Not used in our implementation. Defaults to None. 

        Returns:
            torch.Tensor: Composed weights.
        """
        if scaling is None:
            scaling = self.scaling

        match self.mode:
            case "attention":
                return weights + (self.rescale(added, sigma=self.sigma) * scaling)
            case "dense_fan_in" | "dense_fan_out": 
                return weights * (added * scaling)
            case _:
                return weights

    def com_inv(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Inverts the composition operation between existing and injected weights.

        Args:
            weights (torch.Tensor): Existing weights.
            added (torch.Tensor): Weights to subtract.

        Returns:
            torch.Tensor: Inverted weights.
        """
        match self.mode:
            case "attention":
                return weights - (added * self.scaling)
            case "dense_fan_in" | "dense_fan_out":
                return weights / (added * self.scaling)
            case _:
                return weights

        
    def forward(self, hidden_states: Optional[torch.Tensor], layer_input: torch.Tensor):
        """Forward pass of the LoRA module.
    
        Args:
            hidden_states (Optional[torch.Tensor]): Input tensor for hidden states.
            layer_input (torch.Tensor): Input tensor for the current layer.
    
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Processed hidden states and gate (if applicable).
        """

        # Rescale weights if required
        #
        # Rescaling after the update to the gradients at the
        # end of the epoch is the correct way to do it, but
        # to keep the code within the adapter module, we do it here.
        # So we update the weights in the beginning of the first
        # training step of the next epoch.
        #
        if self._do_rescale():
            self._rescale_weights()

        if self.mode == "attention":
            # If hidden_states is None, use layer_input instead
            if hidden_states is None:
                hidden_states = layer_input
           
            hidden_states = self.dropout(torch.nan_to_num(hidden_states))
            dw = self.f(hidden_states) @ torch.t(self.lora_A) @ torch.t(self.lora_B)
            # Normalize delta_w by its L2 norm
            dw_norm = dw.norm(p=2, dim=1, keepdim=True)
            dw_norm = dw_norm + (dw_norm == 0).float() * 1e-9  # Avoid division by zero
            hidden_states = dw / dw_norm
            
        # Alternative calculation mode
        elif self.mode == "dense_fan_in" or self.mode == "dense_fan_out":
            # Create scaling vector from lora_C and repeat it across batch size
            scaling_vector = torch.nan_to_num(self.lora_C.view(1, 1, -1).repeat(layer_input.shape[0], 1, 1))
            # Apply scaling to the weights
            hidden_states = scaling_vector * (1.0 - self.scalar_scaler)
                  
        # No operation mode
        else:
            # If hidden_states is None, use layer_input instead
            if hidden_states is None:
                hidden_states = layer_input

        self.delta_w = hidden_states
        # Apply gating mechanism if use_gating is enabled
        if self.use_gating:
            # Compute gate values using a sigmoid function applied to the layer input
            gate = torch.sigmoid(self.gate(layer_input))
            # Average gate values across the second dimension and add a new dimension at the end
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            # Multiply hidden_states by the gate values
            hidden_states = hidden_states * gate
        else:
            gate = None

        # Return the processed hidden_states and gate
        return hidden_states, gate
    

class IA3(nn.Module):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
        gating_heads: int = 1,
        location_key: str = None,
    ):
        super().__init__()
        assert config.composition_mode == "scale", "IA3 module only supports composition_mode='scale'."
        if config.r > 1:
            raise ValueError("Can only use composition_mode='scale' when r == 1.")
        self.r = config.r
        self.lora_alpha = config.alpha
        self.composition_mode = config.composition_mode
        self.attn_matrices = config.attn_matrices
        self.use_gating = config.use_gating
        # Optional dropout
        if config.dropout > 0.0:
            raise ValueError("IA3 module does not support dropout.")

        # Actual trainable parameters
        self.lora_B = nn.Parameter(torch.zeros(lora_B_shape))
        self.scaling = self.lora_alpha

        # For compatibility with LoRA, allow all init_weights types here.
        # Usually should be "ia3".
        if config.init_weights == "lora":
            logger.warning("(IA)^3 module initialized with LoRA zeo init. Ignore if this is intended.")
            nn.init.zeros_(self.lora_B)
        elif config.init_weights == "bert":
            nn.init.normal_(self.lora_B, std=0.02)
        elif config.init_weights == "ia3":
            nn.init.ones_(self.lora_B)
        else:
            raise ValueError("Unknown init_weights type: {}".format(config.init_weights))

        if self.use_gating:
            self.gate = nn.Linear(lora_A_shape[-1], gating_heads)
            nn.init.normal_(self.gate.weight, std=0.02)

    @property
    def delta_w(self) -> torch.Tensor:
        return self.lora_B

    def com(self, weights: torch.Tensor, added: torch.Tensor, scaling=None) -> torch.Tensor:
        """Performs the composition operation between existing and injected weights."""
        if scaling is None:
            scaling = self.scaling
        return weights * (added * scaling)

    def com_inv(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Inverts the composition operation between existing and injected weights."""
        return weights / (added * self.scaling)

    def forward(self, hidden_states: Optional[torch.Tensor], layer_input: torch.Tensor):
        scaling_vector = self.lora_B.view(1, 1, -1).repeat(layer_input.shape[0], 1, 1)
        if hidden_states is None:
            hidden_states = scaling_vector
        else:
            hidden_states = hidden_states * scaling_vector
        if self.use_gating:
            gate = torch.sigmoid(self.gate(layer_input))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            hidden_states = hidden_states * gate
        else:
            gate = None

        return hidden_states, gate


class LoRALayer(AdapterLayerBase):
    adapter_modules_name = "loras"

    def __init__(
        self, location_key: str, model_config: PretrainedConfig, adapters_config: ModelAdaptersConfig, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.location_key = location_key + "_lora"
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.loras = nn.ModuleDict(dict())
        self.merged = False

    def get_n_heads(self, lora: Union[LoRA, IA3, LoRAConfig]):
        return 1

    def _check_lora_location(self, config: LoRAConfig):
        return True

    def _get_lora_shapes(self, config: LoRAConfig):
        raise NotImplementedError()

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        lora_config = self.adapters_config.match(
            adapter_name,
            config_type=LoRAConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if lora_config is not None and self._check_lora_location(lora_config):
            if lora_config.composition_mode == "add":
                lora_cls = LoRA
            elif lora_config.composition_mode == "scale":
                lora_cls = IA3
            else:
                raise ValueError(f"Unknown composition_mode: {lora_config.composition_mode}")
            lora = lora_cls(
                *self._get_lora_shapes(lora_config),
                lora_config,
                gating_heads=self.get_n_heads(lora_config),
                location_key=self.location_key,
            )
            lora.train(self.training)
            lora = lora.to(self.weight.device)
            self.loras[adapter_name] = lora
            return True

        return False

    def average_adapter(self, adapter_name: str, input_adapters: Dict[str, float]) -> bool:
        # add new adapter
        if self.add_adapter(adapter_name, self.layer_idx):
            # average weights
            avg_state_dict = {}
            for name, weight in input_adapters.items():
                if name in self.loras:
                    module = self.loras[name]
                    for k, v in module.state_dict().items():
                        if k in avg_state_dict:
                            avg_state_dict[k] += weight * v
                        else:
                            avg_state_dict[k] = weight * v
                else:
                    self.delete_adapter(adapter_name)  # clean up before raising error
                    raise ValueError("Adapter {} not found.".format(name))
            # load averaged weights
            self.loras[adapter_name].load_state_dict(avg_state_dict)
            return True

        return False

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.loras:
            del self.loras[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to lora

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to lora

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        if unfreeze_adapters:
            for name in adapter_setup.flatten():
                if name in self.loras:
                    for param in self.loras[name].parameters():
                        param.requires_grad = True

    def freeze_adapter(self, adapter_name: str, freeze: bool = True):
        if adapter_name in self.loras:
            self.loras[adapter_name].train(not freeze)
            for param in self.loras[adapter_name].parameters():
                param.requires_grad = not freeze

    def get_adapter(self, adapter_name: str) -> nn.Module:
        if adapter_name in self.loras:
            return self.loras[adapter_name]
        else:
            return None


class LoRAState(NamedTuple):
    """Models the input and output states of a LoRA layer.

    Args:
        layer_input (torch.Tensor): The input states to the adapted layer.
        hidden_states (Optional[torch.Tensor]):
            The hidden states of the adaptation module. These can be None before passing through the first LoRA/ IA3
            module.
        layer_output (torch.Tensor): The output states of the original layer without adaptation.
        last (str, optional): Name of the last adapter applied in the composition.
    """

    layer_input: torch.Tensor
    hidden_states: Optional[torch.Tensor]
    layer_output: torch.Tensor
    last: Optional[str]


class LoRALinear(LoRALayer, ComposableAdapterLayerBase):
    """
    LoRA implementation for Linear layer. This layer supports composition.

    Args:
        fan_in_fan_out (bool, optional):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). Defaults to False.
        no_init_bias (bool, optional): Use this to add a bias that is not initialized by PyTorch. Defaults to False.

    """

    supported_compositions = [Stack, BatchSplit, Average, Parallel]
    allow_multi_parallelize = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        attn_key: str = None,
        fan_in_fan_out: bool = False,
        no_init_bias: bool = False,
        **kwargs
    ):
        if no_init_bias and "bias" not in kwargs:
            kwargs["bias"] = False
        LoRALayer.__init__(self, location_key, model_config, adapters_config, in_features, out_features, **kwargs)

        self.attn_key = attn_key
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = torch.t(self.weight.data)
        if no_init_bias:
            self.bias = nn.Parameter(torch.empty(out_features))

    @classmethod
    def wrap(
        cls,
        module: Union[nn.Linear, Conv1D],
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        attn_key: str = None,
        **kwargs
    ):
        if isinstance(module, Conv1D):
            new_module = LoRALinearTorch(
                module.weight.shape[0],
                module.weight.shape[1],
                location_key,
                model_config,
                adapters_config,
                attn_key=attn_key,
                **kwargs,
            )
        else:
            if bitsandbytes_available and isinstance(module, Linear4bit):
                cls = LoRALinear4bit
            elif bitsandbytes_available and isinstance(module, Linear8bitLt):
                cls = LoRALinear8bitLt
            else:
                cls = LoRALinearTorch
            # Make sure that the bias is not added if the original module does not have one
            if "bias" not in kwargs:
                kwargs["bias"] = hasattr(module, "bias") and module.bias is not None
            new_module = cls(
                module.in_features,
                module.out_features,
                location_key,
                model_config,
                adapters_config,
                attn_key=attn_key,
                **kwargs,
            )
        new_module.copy_from(module)

        return new_module

    def copy_from(self, module: nn.Linear):
        self.weight = module.weight
        if module.bias is not None:
            self.bias = module.bias

    def _check_lora_location(self, config: LoRAConfig):
        return self.attn_key is None or self.attn_key in config.attn_matrices

    def _get_lora_shapes(self, config: LoRAConfig):
        return (config.r, self.in_features), (self.out_features, config.r)

    def maybe_t(self, w):
        return torch.t(w) if self.fan_in_fan_out else w

    def merge_adapter(self, name: str):
        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                if lora.use_gating:
                    raise ValueError("Cannot merge LoRA layer with gating.")
                delta_w = self.maybe_t(lora.delta_w)
                self.weight.data = lora.com(self.weight.data, delta_w)
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")

    def reset_adapter(self):
        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            delta_w = self.maybe_t(lora.delta_w)
            self.weight.data = lora.com_inv(self.weight.data, delta_w)
            self.merged = None

    def vslice(self, state: LoRAState, slice_obj: slice) -> LoRAState:
        return LoRAState(
            state.layer_input[slice_obj],
            state.hidden_states[slice_obj] if state.hidden_states is not None else None,
            state.layer_output[slice_obj],
            state.last,
        )

    def pad_and_concat(self, states: List[LoRAState]) -> LoRAState:
        return LoRAState(
            torch.cat([s.layer_input for s in states], dim=0),
            torch.cat([s.hidden_states for s in states], dim=0) if states[0].hidden_states is not None else None,
            torch.cat([s.layer_output for s in states], dim=0),
            states[-1].last,
        )

    def repeat(self, state: LoRAState, channels: int) -> LoRAState:
        return LoRAState(
            state.layer_input.repeat(channels, 1, 1),
            state.hidden_states.repeat(channels, 1, 1) if state.hidden_states is not None else None,
            state.layer_output.repeat(channels, 1, 1),
            state.last,
        )

    def mean(self, states: List[LoRAState], weights: torch.Tensor) -> LoRAState:
        return LoRAState(
            states[0].layer_input,
            torch.mean(torch.stack([s.hidden_states for s in states], dim=0) * weights, dim=0)
            if states[0].hidden_states is not None
            else None,
            states[0].layer_output,
            states[-1].last,
        )

    def compose_single(self, adapter_setup: str, state: LoRAState, lvl: int = 0) -> LoRAState:
        lora = self.loras[adapter_setup]
        hidden_states, gate = lora(state.hidden_states, state.layer_input)
        if gate is not None:
            self._store_gating_score(adapter_setup, gate)

        return state._replace(hidden_states=hidden_states, last=adapter_setup)

    def forward(self, input_states: torch.Tensor):
        if self.fan_in_fan_out:
            weight = torch.transpose(self.weight, -2, -1) if self.fan_in_fan_out else self.weight
            # result shape: <batch_size> x <seq_len> x <head_dim>
            layer_output = F.linear(input_states, weight, bias=self.bias)
        else:
            layer_output = super().forward(input_states)

        if not self.merged:
            adapter_setup = self.get_active_setup()
            if adapter_setup is not None:
                state = LoRAState(input_states, None, layer_output, None)
                state = self.compose(adapter_setup, state)
                _, hidden_states, layer_output, last = state

                last_lora = self.loras[last]
                layer_output = last_lora.com(
                    layer_output, hidden_states, scaling=1.0
                )  # scaling already applied in compose

        return layer_output


class LoRALinearTorch(LoRALinear, nn.Linear):
    pass


if bitsandbytes_available:

    class LoRALinear4bit(LoRALinear, Linear4bit):
        def copy_from(self, module: Linear4bit):
            self.weight = module.weight
            if module.bias is not None:
                self.bias = module.bias
            self.compute_dtype = module.compute_dtype
            self.compute_type_is_set = module.compute_type_is_set
            self.quant_state = module.quant_state
            self.quant_storage = module.quant_storage

        def merge_adapter(self, name: str):
            if name in self.loras:
                if self.merged == name:
                    return  # already merged
                elif not self.merged:
                    lora = self.loras[name]
                    if lora.use_gating:
                        raise ValueError("Cannot merge LoRA layer with gating.")
                    delta_w = self.maybe_t(lora.delta_w)
                    layer_weight = dequantize_bnb_weight(self.weight, state=self.quant_state)
                    kwargs = self.weight.__dict__
                    merged_weight = lora.com(layer_weight, delta_w)
                    self.weight = Params4bit(merged_weight.to("cpu"), requires_grad=False, **kwargs).to(
                        self.weight.device
                    )
                    self.merged = name
                elif self.merged != name:
                    raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")

        def reset_adapter(self):
            if self.merged:
                lora = self.loras[self.merged]
                delta_w = self.maybe_t(lora.delta_w)
                merged_weight = dequantize_bnb_weight(self.weight, state=self.quant_state)
                kwargs = self.weight.__dict__
                layer_weight = lora.com_inv(merged_weight, delta_w)
                self.weight = Params4bit(layer_weight.to("cpu"), requires_grad=False, **kwargs).to(self.weight.device)
                self.merged = None

    class LoRALinear8bitLt(LoRALinear, Linear8bitLt):
        def copy_from(self, module: Linear8bitLt):
            self.weight = module.weight
            if module.bias is not None:
                self.bias = module.bias
            self.state = module.state
            self.index = module.index

        def merge_adapter(self, name: str):
            if name in self.loras:
                if self.merged == name:
                    return  # already merged
                elif not self.merged:
                    lora = self.loras[name]
                    if lora.use_gating:
                        raise ValueError("Cannot merge LoRA layer with gating.")
                    delta_w = self.maybe_t(lora.delta_w)
                    layer_weight = dequantize_bnb_weight(self.weight, state=self.state)
                    merged_weight = lora.com(layer_weight, delta_w)
                    self.weight = Int8Params(
                        merged_weight.to("cpu"), requires_grad=False, has_fp16_weights=self.weight.has_fp16_weights
                    ).to(self.weight.device)
                    self.state.reset_grads()
                    self.merged = name
                elif self.merged != name:
                    raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")

        def reset_adapter(self):
            if self.merged:
                lora = self.loras[self.merged]
                delta_w = self.maybe_t(lora.delta_w)
                merged_weight = dequantize_bnb_weight(self.weight, state=self.state)
                layer_weight = lora.com_inv(merged_weight, delta_w)
                self.weight = Int8Params(
                    layer_weight.to("cpu"), requires_grad=False, has_fp16_weights=self.weight.has_fp16_weights
                ).to(self.weight.device)
                self.state.reset_grads()
                self.merged = None


class LoRAMergedLinear(LoRALayer, nn.Linear):
    """
    LoRA implementation for merged attention layer, as used by some model implementations (e.g. GPT-2). This layer
    currently does not support composition.

    Args:
        fan_in_fan_out (bool, optional):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). Defaults to False.
        no_init_bias (bool, optional): Use this to add a bias that is not initialized by PyTorch. Defaults to False.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        fan_in_fan_out: bool = False,
        no_init_bias: bool = False,
        **kwargs
    ):
        if no_init_bias and "bias" not in kwargs:
            kwargs["bias"] = False
        LoRALayer.__init__(self, location_key, model_config, adapters_config, in_features, out_features, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        if no_init_bias:
            self.bias = nn.Parameter(torch.empty(out_features))

    @classmethod
    def wrap(
        cls,
        module: Union[nn.Linear, Conv1D],
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        **kwargs
    ):
        if isinstance(module, Conv1D):
            new_module = cls(
                module.weight.shape[0], module.weight.shape[1], location_key, model_config, adapters_config, **kwargs
            )
        else:
            new_module = cls(
                module.in_features, module.out_features, location_key, model_config, adapters_config, **kwargs
            )
        new_module.weight = module.weight
        if module.bias is not None:
            new_module.bias = module.bias

        return new_module

    def get_n_heads(self, lora: Union[LoRA, IA3, LoRAConfig]):
        return len(set(lora.attn_matrices))

    def _get_lora_shapes(self, config: LoRAConfig):
        n_heads = self.get_n_heads(config)
        return (config.r * n_heads, self.in_features), (
            self.out_features // 3 * n_heads,
            config.r,
        )

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        is_added = super().add_adapter(adapter_name, layer_idx)
        if is_added:
            lora_config = lora_config = self.adapters_config.match(
                adapter_name,
                config_type=LoRAConfig,
                layer_idx=self.layer_idx,
                location_key=self.location_key,
            )
            lora = self.loras[adapter_name]
            lora.enable_lora = [
                "q" in lora_config.attn_matrices,
                "k" in lora_config.attn_matrices,
                "v" in lora_config.attn_matrices,
            ]
            # Actual trainable parameters
            if any(lora.enable_lora):
                # Compute the indices
                lora.lora_ind = self.weight.new_zeros((self.out_features,), dtype=torch.bool).view(
                    len(lora.enable_lora), -1
                )
                lora.lora_ind[lora.enable_lora, :] = True
                lora.lora_ind = lora.lora_ind.view(-1)
            return True
        else:
            return False

    def pad(self, x, lora, fill_value=None):
        if fill_value is None:
            if lora.composition_mode == "add":
                fill_value = 0
            else:
                fill_value = 1
        result = x.new_full((*x.shape[:-1], self.out_features), fill_value)
        result = result.view(-1, self.out_features)
        result[:, lora.lora_ind] = x.reshape(-1, self.out_features // 3 * self.get_n_heads(lora))
        return result.view((*x.shape[:-1], self.out_features))

    def reset_adapter(self):
        def T(w):
            return w if self.fan_in_fan_out else torch.t(w)

        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            if lora.r > 0 and any(lora.enable_lora):
                if lora.composition_mode == "scale":
                    delta_w = lora.lora_B
                else:
                    delta_w = F.conv1d(
                        lora.lora_A.data.unsqueeze(0), lora.lora_B.data.unsqueeze(-1), groups=sum(lora.enable_lora)
                    ).squeeze(0)
                # shape after transpose: <head_dim> x <head_dim * n_heads>
                delta_w = delta_w.transpose(-2, -1)
                self.weight.data = lora.com_inv(self.weight.data, T(self.pad(delta_w, lora)))
            self.merged = None

    def _compute_adapted_weight(self, name, lora):
        def T(w):
            return w if self.fan_in_fan_out else torch.t(w)

        weight = self.weight
        if lora.r > 0:
            if lora.composition_mode == "scale":
                delta_w = lora.lora_B
            else:
                delta_w = F.conv1d(
                    lora.lora_A.data.unsqueeze(0), lora.lora_B.data.unsqueeze(-1), groups=sum(lora.enable_lora)
                ).squeeze(0)
            # shape after transpose: <head_dim> x <head_dim * n_heads>
            delta_w = delta_w.transpose(-2, -1)
            weight = lora.com(weight, T(self.pad(delta_w, lora)))

        return weight

    def merge_adapter(self, name: str):
        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                if lora.use_gating:
                    raise ValueError("Cannot merge LoRA layer with gating.")
                self.weight.data = self._compute_adapted_weight(name, lora)
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRALayer already has a merged LoRA module. Please reset it first.")

    def forward(self, x: torch.Tensor):
        def T(w):
            return torch.t(w) if self.fan_in_fan_out else w

        if not self.merged:
            adapter_setup = self.get_active_setup()
            if adapter_setup is not None:
                if len(adapter_setup) == 1:
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    lora = self.loras[adapter_setup[0]]
                    if lora.r > 0:
                        if lora.composition_mode == "scale":
                            delta_w = lora.lora_B.view(1, 1, -1)
                        else:
                            after_A = F.linear(lora.lora_dropout(x), lora.lora_A)
                            after_B = F.conv1d(
                                after_A.transpose(-2, -1), lora.lora_B.unsqueeze(-1), groups=sum(lora.enable_lora)
                            ).transpose(-2, -1)
                            delta_w = after_B
                        if lora.use_gating:
                            gate = torch.sigmoid(lora.gate(x))
                            gate = torch.mean(gate, dim=1)
                            self._store_gating_score(adapter_setup[0], gate)
                            gate = self.pad(
                                gate.repeat_interleave(self.out_features // 3, dim=-1), lora, fill_value=1
                            ).unsqueeze(1)
                        else:
                            gate = None
                        # result = (batch_size, seq_len, head_dim * 3)
                        result = lora.com(result, self.pad(delta_w, lora), scaling=gate)
                    return result
                else:
                    raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with LoRA.")

        return F.linear(x, T(self.weight), bias=self.bias)
