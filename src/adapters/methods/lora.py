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
import matplotlib.pyplot as plt

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


class LoRA(nn.Module):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
        gating_heads: int = 1,
        location_key: str = None,
    ):
        super().__init__() 
        # Ensure the composition mode is 'add'
        assert config.composition_mode == "add", "LoRA module only supports composition_mode='add'."
       
        # Initialize configuration parameters
        
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
        self.autoencoder_sigmas = None
        self.A_sigma = None
        self.B_sigma = 0.0
        self.composition_mode = config.composition_mode
        self.attn_matrices = config.attn_matrices
        self.use_gating = config.use_gating
        self.non_linearity = config.non_linearity 
        self.sigma = None
        self._delta_w = None  # Placeholder for delta weights
        # List to store variance for each LoRA instance
        
        self.dropout = nn.Dropout(p=config.dropout) if config.dropout > 0.0 else lambda x: x
        
        self.location = self._get_valid_location_key(config, location_key)
        self.variances = {self.location+"_W":[], self.location+"_delta_w": []}

        
        self._layer_specific_setup(lora_A_shape, lora_B_shape)
        # Setup gating mechanism if required
        self._setup_gating_maybe(gating_heads)
        self.batches_per_epoch = self._calculate_batches_per_epoch(config.batch_size, config.training_set_size)
        self.n_batches = 0 # have not trained yet   
        self.training_steps = 0
        self.sigma_w = 0.0
        self.epoch = 1

    def _calculate_batches_per_epoch(self, batch_size: Optional[int], training_set_size: Optional[int]) -> int:
        """
        Calculates the number of batches per epoch based on the batch size and training set size.
        """
        if batch_size is not None and training_set_size is not None:
            batches_per_epoch = training_set_size // batch_size
            if batches_per_epoch < 1:
                logging.warning("Turning off rescaling...")
            return batches_per_epoch
        
        logging.warning("Batch size or training set size is None. \
                        Cannot calculate batches per epoch. Setting to 1. \
                        This may lead to incorrect rescaling and suboptimal performance.")
        return 1
            
    def _get_valid_location_key(self, config, location_key) -> bool:
        """
        Checks if the location key is valid based on the configuration.
        """
        match location_key:
            case "selfattn_lora" if config.selfattn_lora:
                if self.connections_in != self.connections_out:
                    logging.warning("Self-attention requires connections_in == connections_out!")
                return "selfattn"
            case "intermediate_lora" if config.intermediate_lora:
                if self.connections_in >= self.connections_out:
                    logging.warning("Intermediate requires connections_in < connections_out!")
                return "intermediate"
            case "output_lora" if config.output_lora:
                if self.connections_in <= self.connections_out:
                    logging.warning("Output requires connections_in > connections_out!")
                return "output"
            case _:
                raise ValueError(f"Invalid location key: {location_key}")
            
    def _get_autoencoder_architecture(self, arch: str = "NLbLN") -> nn.Sequential:
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
        
    
    def _layer_specific_setup(self, lora_A_shape, lora_B_shape):
         # Determine calculation mode and setup accordingly
        match self.location:
            case "selfattn":
                self._setup_in_attn(lora_A_shape=lora_A_shape, lora_B_shape=lora_B_shape)
            case "output" | "intermediate":
                self._setup_scaling()
            case _:
                pass

    def _get_neg_slope(self, non_linearity: str = "leakyrelu") -> float:
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

            
    def _setup_gating_maybe(self, gating_heads: int):
        """
        Sets up the gating mechanism if use_gating is enabled.

        Args:
            gating_heads (int): Number of gating heads.
        """
        if self.use_gating:
            self.gate = nn.Linear(self.connections_in, gating_heads, dtype=torch.float32)
            nn.init.normal_(self.gate.weight, std=0.02)

    def _setup_scaling(self):
        """
        Sets up the basic calculation mode by initializing scaling parameters.
        """
        self.lora_C = nn.Parameter(torch.ones(self.connections_out, 1, dtype=torch.float32))
        self.scalar_scaler = nn.Parameter(torch.tensor(1e-9, dtype=torch.float32))
        sigma = self._estimate_scaling_sigma()
        nn.init.normal_(self.lora_C, mean=1.0, std=sigma)
        self.sigma = self.lora_C.std().item()
        self.variances[self.location+"_lora_C"] = [self.lora_C.var().item()]

    def _estimate_scaling_sigma(self) -> float:
        return math.sqrt(2 / ((1 + (self._get_neg_slope(self.non_linearity)) ** 2) * self.connections_out))
    
    def _estimate_attn_sigma(self, tensor: torch.Tensor, mode: Literal["fan_in", "fan_out"] = "fan_in"):
        fan = nn.init._calculate_correct_fan(tensor, mode=mode)
        gain = nn.init.calculate_gain("leaky_relu", param=math.sqrt(5))
        sigma = gain * math.sqrt(2.0 / float(fan))
        return sigma
            
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
        self.sigma = self.A_sigma
        
        

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
        #self.A_sigma = self._estimate_attn_sigma(self.lora_A.data, mode="fan_in")
        self.A_sigma = self.lora_A.std().item()
        self.variances[self.location+"_lora_A"] = [self.lora_A.var().item()]
        nn.init.zeros_(self.lora_B)
        #self.B_sigma = 0.0
        self.B_sigma = self.lora_B.std().item()
        self.variances[self.location+"_lora_B"] = [self.lora_B.var().item()]

    def _initialize_autoencoder_weights(self, layers: nn.Sequential):
        """
        Initializes the weights of the given layers.
        
        Args:
            layers (nn.Sequential): Sequential model containing the layers.
        """
        self.autoencoder_sigmas = torch.zeros(len(layers), dtype=torch.float32)
        # fan in for encoder, fan out for decoder
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                if i < len(layers) / 2:
                    mode = "fan_in"
                else:
                    mode = "fan_out"
                
                nn.init.kaiming_normal_(layer.weight, mode=mode, a=math.sqrt(5))
                # sigma = self._estimate_attn_sigma(layer.weight, mode=mode)
                sigma = layer.weight.std().item()
                self.autoencoder_sigmas[i] = sigma
                self.variances[f"{self.location}_autoencoder_{i}"] = [layer.weight.var().item()]
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    
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

    def _increment_training_step_maybe(self):
        if self.training:
            self.training_steps = self.training_steps + 1
            self.n_batches = self.n_batches + 1
            if self.n_batches > self.batches_per_epoch:
                self.n_batches = 1
                self.epoch = self.epoch + 1

    def _epoch_start(self) -> bool:
        """
        Checks if rescaling is required based on the configuration.

        Returns:
            bool: True if rescaling is required, False otherwise.
        """
        if not self.training:
            return False
        
        if self.batches_per_epoch < 1:
            return False
        
        if not self.training:
            return False
        
        if self.n_batches  == 1: 
            return True
        return False
    
    def _epoch_end(self) -> bool:
        if not self.training:
            return False
        
        if self.batches_per_epoch < 1:
            return False
        
        if not self.training:
            return False
        
        if self.n_batches == self.batches_per_epoch:
            return True
        return False
    
    def _rescale_autoencoder_weights(self):
        """
        Rescales the weights of the autoencoder.
        """
        for layer, sigma in zip(self.f, self.autoencoder_sigmas):
            if isinstance(layer, nn.Linear):
                assert sigma, "Sigma must be set."
                
                layer.weight.data = self.rescale(layer.weight.data, sigma=sigma)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            
    def rescale_weights_maybe(self):
        """
        Rescale the weights based on the current configuration.
        """
        if not self._epoch_start() or self.epoch == 1:
            return
        
        if self.location in ["output", "intermediate"]:
            self.lora_C.data = self.rescale(self.lora_C.data, sigma=self.sigma, dtype=torch.float32)
        elif self.location == "selfattn":
            self.lora_A.data = self.rescale(self.lora_A.data, sigma=self.A_sigma)
        #    self._rescale_autoencoder_weights()
        
            
    def record_weights_var_maybe(self) -> None:
        """
        Calculates the variance of the given weights.

        Args:
            weights (torch.Tensor): Weights to calculate the variance for.

        Returns:
            float: Variance of the weights.
        """
        if self.training:
            with torch.no_grad():
                if self.location == "selfattn":
                    self.variances[self.location+"_lora_A"].append(self.lora_A.var().item())
                    self.variances[self.location+"_lora_B"].append(self.lora_B.var().item())
                    for i, layer in enumerate(self.f):
                        if isinstance(layer, nn.Linear):
                            self.variances[f"{self.location}_autoencoder_{i}"].append(torch.var(layer.weight).item())
                else:
                    self.variances[self.location+"_lora_C"].append(torch.var(self.lora_C).item())
        
    def record_dw_var_maybe(self, hidden_states: torch.Tensor) -> None:
        if self.training:
            with torch.no_grad():
                self.variances[self.location+"_delta_w"].append(hidden_states.var().item())
    
    def record_w_var_maybe(self, weights: torch.Tensor) -> None:
        if self.training:
            with torch.no_grad():
                self.variances[self.location+"_W"].append(weights.var().item())

    def rescale(self, weights: torch.Tensor, sigma: float = 0.05, dtype: torch.dtype = None) -> torch.Tensor:
        if sigma == 0:
            return weights
        w = torch.nan_to_num(weights)
        u = torch.mean(w, dtype=dtype)
        stddev = torch.std(w)
  
        # calculate z-scores
        z = (w - u) / (stddev + 1e-12)
        # rescale to original range
        return z * sigma + u
    
    def get_variances(self) -> Dict[str, List[float]]:
        """
        Returns the recorded variances for each parameter.

        Returns:
            Dict[str, List[float]]: Dictionary with variance lists for each parameter.
        """
        return self.variances
    
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
        if self.training and self.training_steps == 1:
            self.sigma_w = weights.std().item()
        # burn in period
        
        if self._epoch_start() and self.epoch > 1:
            w = self.rescale(weights, self.sigma_w)
        else:
            w = weights

        if scaling is None:
            scaling = self.scaling

        self.record_dw_var_maybe(added * scaling)
        self.record_w_var_maybe(w)
        self.record_weights_var_maybe()
        match self.location:
            case "selfattn":
                return w + added * scaling
            case "output" | "intermediate": 
                return w * (added * scaling)
            case _:
                return w
    

    def com_inv(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Inverts the composition operation between existing and injected weights.

        Args:
            weights (torch.Tensor): Existing weights.
            added (torch.Tensor): Weights to subtract.

        Returns:
            torch.Tensor: Inverted weights.
        """
 
        match self.location:
            case "selfattn":
                return weights - (added * self.scaling)
            case "output" | "intermediate":
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
        self._increment_training_step_maybe()
        self.rescale_weights_maybe()
        self.record_weights_var_maybe()
        # if self._epoch_start():
        #if self._epoch_start():
        #    self.rescale_weights()
        
        if self.location == "selfattn":
            # If hidden_states is None, use layer_input instead
            if hidden_states is None:
                hidden_states = layer_input
            x = torch.nan_to_num(hidden_states)
            self.record_dw_var_maybe(x)
            fx = self.f(self.dropout(x))
            dw = fx @ torch.t(self.lora_A) @ torch.t(self.lora_B)
            # Normalize delta_w by its L2 norm
            dw_norm = dw.norm(p=2, dim=1, keepdim=True) + 1e-9
            normed_dw = dw / dw_norm
            if normed_dw.std() > self.sigma:
                hidden_states = self.rescale(normed_dw, self.sigma)  
            else:
                hidden_states = normed_dw     
            
        # scaling mode
        else:
            # Create scaling vector from lora_C and repeat it across batch size
            scaling_vector = torch.nan_to_num(self.lora_C.view(1, 1, -1).repeat(layer_input.shape[0], 1, 1))
            self.record_dw_var_maybe(scaling_vector)
            hidden_states = scaling_vector * (1.0 - self.scalar_scaler) 
            

        self.delta_w = hidden_states.clone()
        

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
                            avg_state_dict[k] = avg_state_dict[k] + weight * v
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
