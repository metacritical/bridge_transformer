import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

class BridgeAdapter(nn.Module):
    """
    Bridge adapter that can be added to any pretrained transformer model
    to enable connection to external knowledge sources.
    """
    
    def __init__(
        self, 
        base_model: PreTrainedModel, 
        tokenizer,
        hidden_size: int = 768,
        bridge_size: int = 64,
        bridge_layers: List[int] = None,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        freeze_base_model: bool = True
    ):
        super().__init__()
        
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.bridge_size = bridge_size
        
        # Determine which layers to add bridge to
        if bridge_layers is None:
            # Use 1/3 and 2/3 of the way through the model by default
            num_layers = self.base_model.config.num_hidden_layers
            self.bridge_layers = [num_layers // 3, (2 * num_layers) // 3]
        else:
            self.bridge_layers = bridge_layers
        
        # Freeze base model if requested
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Add LoRA adapters to base model for fine-tuning
        if lora_rank > 0:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
            self.base_model.print_trainable_parameters()
        
        # Bridge components
        self.bridge_detectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, bridge_size),
                nn.GELU(),
                nn.Linear(bridge_size, 1),
                nn.Sigmoid()
            ) for _ in self.bridge_layers
        ])
        
        # Query encoder to convert hidden states to query vector
        self.query_encoder = nn.Sequential(
            nn.Linear(hidden_size, bridge_size * 2),
            nn.GELU(),
            nn.Linear(bridge_size * 2, bridge_size)
        )
        
        # Response integration to incorporate external knowledge
        self.response_integrator = nn.Sequential(
            nn.Linear(bridge_size, bridge_size * 2),
            nn.GELU(),
            nn.Linear(bridge_size * 2, hidden_size)
        )
        
        # Bridge activation threshold
        self.bridge_threshold = 0.8
    
    def _get_layer_output_hook(self, layer_idx):
        """
        Returns a hook function that saves the output of a specific layer
        """
        def hook(module, input, output):
            self.layer_outputs[layer_idx] = output
        return hook
    
    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        labels=None,
        external_info=None,
        output_bridge_info=False,
        **kwargs
    ):
        """
        Forward pass with bridge functionality
        """
        # Store layer outputs
        self.layer_outputs = {}
        hooks = []
        
        # Register hooks to capture layer outputs
        for layer_idx in self.bridge_layers:
            # The exact layer access depends on model architecture
            try:
                # For models with .layers attribute (like GPT-2)
                layer = self.base_model.base_model.transformer.h[layer_idx]
            except (AttributeError, IndexError):
                try:
                    # For models with .layers attribute
                    layer = self.base_model.base_model.layers[layer_idx]
                except (AttributeError, IndexError):
                    try:
                        # For models with .encoder.layer
                        layer = self.base_model.base_model.encoder.layer[layer_idx]
                    except (AttributeError, IndexError):
                        print(f"Could not locate layer {layer_idx}. Bridge functionality may not work.")
                        continue
            
            hook = layer.register_forward_hook(self._get_layer_output_hook(layer_idx))
            hooks.append(hook)
        
        # First forward pass
        first_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # Don't compute loss yet
            **kwargs
        )
        
        # Process bridge logic
        bridge_activated = False
        bridge_layer_idx = None
        bridge_query = None
        bridge_activation_values = {}
        
        # Check each monitored layer for bridge activation
        for i, layer_idx in enumerate(self.bridge_layers):
            if layer_idx not in self.layer_outputs:
                continue
            
            # Get last token hidden state from this layer
            layer_output = self.layer_outputs[layer_idx]
            
            # For different model architectures, output structure varies
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
            
            # Get last token from sequence for each example in batch
            # Shape: [batch_size, hidden_size]
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_token_states = hidden_states[batch_indices, seq_lengths]
            
            # Detect if bridge should be activated
            bridge_detector = self.bridge_detectors[i]
            activation_value = bridge_detector(last_token_states)
            
            # Store activation values for return if needed
            bridge_activation_values[layer_idx] = activation_value
            
            # Check if bridge activation threshold is met
            if (activation_value > self.bridge_threshold).any():
                bridge_activated = True
                bridge_layer_idx = layer_idx
                
                # Generate query embedding from activated examples
                activated_indices = (activation_value > self.bridge_threshold).nonzero().squeeze(1)
                activated_states = last_token_states[activated_indices]
                bridge_query = self.query_encoder(activated_states)
                
                # Only use first activated layer (could be modified to use multiple)
                break
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # If bridge activated and external info is available, do second pass
        final_outputs = first_outputs
        if bridge_activated and external_info is not None:
            # Integrate external info
            integrated_info = self.response_integrator(external_info)
            
            # For second pass, we'll modify the hidden states of the activated layer
            # This requires custom forward through the model, which depends on architecture
            # Here we'll use a simplified approach by just adding to the logits
            
            # Add integrated info to the logits of the tokens we want to influence
            # This is a simplification; a real implementation would modify internal states
            logits = first_outputs.logits
            last_token_logits = logits[batch_indices, seq_lengths]
            
            # Only for activated examples
            if len(activated_indices) > 0:
                # Convert integrated info to logit space (simplification)
                logit_adjustment = F.linear(integrated_info, self.base_model.get_output_embeddings().weight)
                
                # Apply to last token logits
                last_token_logits[activated_indices] += logit_adjustment
                
                # Update the logits in the output
                logits[batch_indices, seq_lengths] = last_token_logits
                final_outputs.logits = logits
        
        # Compute loss if labels are provided
        if labels is not None:
            # Calculate loss
            shift_logits = final_outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            final_outputs.loss = loss
        
        # Add bridge info to outputs if requested
        if output_bridge_info:
            bridge_info = {
                "activated": bridge_activated,
                "layer_idx": bridge_layer_idx,
                "query": bridge_query,
                "activation_values": bridge_activation_values
            }
            final_outputs.bridge_info = bridge_info
        
        return final_outputs
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        external_service=None,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        **kwargs
    ):
        """
        Generation with bridge functionality
        """
        for _ in range(max_length):
            # Forward pass with bridge detection
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_bridge_info=True,
                **kwargs
            )
            
            # Get logits for next token prediction
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Check if bridge was activated
            bridge_info = outputs.bridge_info
            if bridge_info["activated"] and external_service is not None:
                # Call external service with query
                query = bridge_info["query"]
                external_info = external_service.get_information(
                    query_embedding=query,
                    query_text=self.tokenizer.decode(input_ids[0, -20:]),  # Use recent tokens as text query
                    use_web=True
                )
                
                # Re-run forward pass with external info
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    external_info=external_info,
                    **kwargs
                )
                
                # Update logits
                next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append next token to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Update attention mask if it exists
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    attention_mask.new_ones((attention_mask.shape[0], 1))
                ], dim=-1)
            
            # Check if we've hit the EOS token
            if next_token[0, 0].item() == self.tokenizer.eos_token_id:
                break
                
        return input_ids

# Helper function to load model with bridge adapter
def load_model_with_bridge(
    model_name="Qwen/Qwen-1_8B", 
    bridge_size=64,
    bridge_layers=None,
    lora_rank=8,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load a HuggingFace model with bridge adapter
    """
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get hidden size from model config
    hidden_size = model.config.hidden_size
    
    # Create bridge adapter
    bridge_model = BridgeAdapter(
        base_model=model,
        tokenizer=tokenizer,
        hidden_size=hidden_size,
        bridge_size=bridge_size,
        bridge_layers=bridge_layers,
        lora_rank=lora_rank
    )
    
    # Move to device
    bridge_model = bridge_model.to(device)
    
    return bridge_model, tokenizer