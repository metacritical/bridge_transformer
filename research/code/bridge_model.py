import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Hyperparameters
class BridgeModelConfig:
    def __init__(self):
        self.vocab_size = 50257  # GPT-2 vocabulary size
        self.n_embd = 768  # embedding dimension
        self.n_head = 12  # number of attention heads
        self.n_layer = 12  # number of layers
        self.block_size = 1024  # context length
        self.dropout = 0.1
        self.bridge_neurons_pct = 0.05  # 5% of neurons dedicated as bridge
        self.bridge_activation_threshold = 0.8  # threshold for triggering bridge

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

class BridgeAttention(nn.Module):
    """Multi-head attention with dedicated bridge neurons"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        assert self.n_embd % self.n_head == 0
        
        # key, query, value projections
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)))
        
        # Bridge neurons (a subset of neurons in the output projection)
        self.bridge_size = int(config.bridge_neurons_pct * config.n_embd)
        self.bridge_neurons_idx = torch.randperm(config.n_embd)[:self.bridge_size]
        
        # For detecting bridge activation
        self.bridge_detector = nn.Linear(self.bridge_size, 1)
        self.bridge_activation_threshold = config.bridge_activation_threshold

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        
        # calculate query, key, values
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_dropout(self.proj(y))
        
        # Extract bridge neurons activation
        bridge_activations = y[:, -1, self.bridge_neurons_idx]  # Get last token's bridge neurons
        bridge_signal = torch.sigmoid(self.bridge_detector(bridge_activations))
        
        # Check if bridge is activated
        bridge_activated = bridge_signal > self.bridge_activation_threshold
        
        return y, bridge_activated, bridge_activations

class BridgeMLP(nn.Module):
    """MLP with bridge neuron capabilities"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Bridge neurons in the MLP
        self.bridge_size = int(config.bridge_neurons_pct * config.n_embd)
        self.bridge_neurons_idx = torch.randperm(config.n_embd)[:self.bridge_size]

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class BridgeBlock(nn.Module):
    """Transformer block with bridge capabilities"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = BridgeAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = BridgeMLP(config)

    def forward(self, x):
        attn_output, bridge_activated, bridge_activations = self.attn(self.ln_1(x))
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, bridge_activated, bridge_activations

class BridgeTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([BridgeBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Query encoder for bridge functionality
        self.query_encoder = nn.Linear(config.n_embd, 256)
        
        # Response decoder for integrating external information
        self.response_decoder = nn.Linear(256, config.n_embd)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Bridge layer selection - we'll monitor bridge signals from these layers
        self.bridge_layers = [4, 8]  # Monitor middle and later layers
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, external_response=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Get token and position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        bridge_signals = []
        bridge_queries = []
        
        # Process through transformer blocks
        for i, block in enumerate(self.transformer.h):
            x, bridge_activated, bridge_activations = block(x)
            
            # Store bridge signals from monitored layers
            if i in self.bridge_layers and bridge_activated.any():
                bridge_signals.append((i, bridge_activated))
                
                # Generate query from bridge activations
                query = self.query_encoder(bridge_activations)
                bridge_queries.append(query)
        
        # Apply final layer norm
        x = self.transformer.ln_f(x)
        
        # If external response is provided and bridge was activated, integrate it
        if external_response is not None and len(bridge_queries) > 0:
            # Convert external response to embeddings
            response_embedding = self.response_decoder(external_response)
            
            # Simple integration: add to the output embeddings
            # In real implementation, this would be more sophisticated
            x[:, -1, :] = x[:, -1, :] + response_embedding
        
        # Language modeling logits
        logits = self.lm_head(x)
        
        return logits, bridge_signals, bridge_queries
    
    def generate(self, idx, max_new_tokens, temperature=1.0, external_service=None):
        """
        Generate text with bridge functionality to external knowledge
        """
        for _ in range(max_new_tokens):
            # Crop idx to block_size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass through the model
            logits, bridge_signals, bridge_queries = self.forward(idx_cond)
            
            # Get the last token logits
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # If bridge is activated and external service is available
            if len(bridge_signals) > 0 and external_service is not None:
                # Use the most recent bridge query
                query = bridge_queries[-1]
                
                # Call external service to get information
                external_info = external_service.get_information(query)
                
                # Integrate external info and regenerate token
                logits, _, _ = self.forward(idx_cond, external_info)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# -----------------------------------------------------------------------------
# External Knowledge Service Mock
# -----------------------------------------------------------------------------

class ExternalKnowledgeService:
    """Mock external knowledge service"""
    def __init__(self):
        self.knowledge_base = {
            "capital:france": torch.randn(256),  # Embedding for "Paris is the capital of France"
            "inventor:telephone": torch.randn(256),  # Embedding for "Alexander Graham Bell"
            # Add more knowledge entries as needed
        }
    
    def get_information(self, query):
        """Convert query embedding to closest knowledge entry"""
        best_match = None
        best_score = -float('inf')
        
        # Simple cosine similarity search
        for key, value in self.knowledge_base.items():
            sim = F.cosine_similarity(query, value.unsqueeze(0), dim=1)
            if sim.item() > best_score:
                best_score = sim.item()
                best_match = value
        
        return best_match if best_match is not None else torch.zeros(256)