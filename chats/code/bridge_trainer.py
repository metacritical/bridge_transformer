import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

# Import our model and data modules
from bridge_model import BridgeTransformer, BridgeModelConfig, ExternalKnowledgeService
from data_preparation import LLMDataGenerator, BridgeDataset, create_dataloaders
from real_knowledge_service import RealExternalKnowledgeService

# -----------------------------------------------------------------------------
# Training Utilities
# -----------------------------------------------------------------------------

class BridgeTrainer:
    def __init__(self, model, config, train_loader, val_loader, device='cuda', 
                 save_dir='checkpoints', log_dir='logs', external_service=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        # Create directories if they don't exist
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # External knowledge service
        self.ext_service = external_service or ExternalKnowledgeService()
        
    def train(self, epochs=10, lr=3e-4, warmup_epochs=2, phase='supervised'):
        """
        Train the model with multiple phases:
        - phase='supervised': Basic supervised learning
        - phase='bridge_detection': Focus on bridge activation learning
        - phase='bridge_retrieval': Focus on information retrieval
        - phase='rl': Reinforcement learning phase
        """
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Loss functions
        lm_criterion = nn.CrossEntropyLoss()
        bridge_criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            train_bridge_losses = []
            train_lm_losses = []
            start_time = time.time()
            
            # Learning rate warmup
            if epoch < warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * (epoch + 1) / warmup_epochs
            
            for step, batch in enumerate(self.train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                bridge_labels = batch['bridge_label'].to(self.device)
                output_ids = batch['output_ids'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass based on training phase
                if phase == 'supervised':
                    # Standard language modeling forward pass
                    logits, bridge_signals, _ = self.model(input_ids)
                    
                    # Calculate language modeling loss
                    # Shift logits and targets for next-token prediction
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_targets = input_ids[:, 1:].contiguous()
                    lm_loss = lm_criterion(shift_logits.view(-1, self.config.vocab_size), 
                                          shift_targets.view(-1))
                    
                    # Calculate bridge detection loss if we have bridge signals
                    if bridge_signals:
                        # Extract bridge activation from last layer
                        bridge_preds = torch.cat([b[1] for b in bridge_signals if b[0] == max(self.model.bridge_layers)])
                        bridge_loss = bridge_criterion(bridge_preds.float(), bridge_labels)
                    else:
                        bridge_loss = torch.tensor(0.0, device=self.device)
                    
                    # Combine losses
                    loss = lm_loss + bridge_loss
                    
                elif phase == 'bridge_detection':
                    # Focus on bridge detection with higher weight
                    logits, bridge_signals, _ = self.model(input_ids)
                    
                    # Calculate language modeling loss (reduced weight)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_targets = input_ids[:, 1:].contiguous()
                    lm_loss = lm_criterion(shift_logits.view(-1, self.config.vocab_size), 
                                          shift_targets.view(-1))
                    
                    # Calculate bridge detection loss (higher weight)
                    if bridge_signals:
                        bridge_preds = torch.cat([b[1] for b in bridge_signals if b[0] == max(self.model.bridge_layers)])
                        bridge_loss = bridge_criterion(bridge_preds.float(), bridge_labels) * 3.0  # Higher weight
                    else:
                        bridge_loss = torch.tensor(0.0, device=self.device)
                    
                    loss = lm_loss * 0.5 + bridge_loss  # Emphasize bridge detection
                
                elif phase == 'bridge_retrieval':
                    # Get query representation from batch if available
                    query_reprs = batch.get('query_repr', [None] * input_ids.size(0))
                    
                    # First forward pass to get bridge signals
                    _, bridge_signals, bridge_queries = self.model(input_ids)
                    
                    # If bridge activated, get external info and do second pass
                    if bridge_signals and any(b[1].any() for b in bridge_signals):
                        # Process each example in batch
                        external_infos = []
                        
                        for i, (activated, query) in enumerate(zip(
                            [any(b[1][i].item() for b in bridge_signals) for i in range(input_ids.size(0))],
                            bridge_queries[-1] if bridge_queries else [None] * input_ids.size(0)
                        )):
                            if activated:
                                # Use query representation from data if available, otherwise use model's query
                                query_text = query_reprs[i] if query_reprs[i] else None
                                external_info = self.ext_service.get_information(query, query_text=query_text)
                                external_infos.append(external_info)
                            else:
                                external_infos.append(None)
                        
                        # Second forward pass with retrieved info
                        logits, _, _ = self.model(input_ids, external_info=external_infos)
                    else:
                        # Regular forward pass
                        logits, _, _ = self.model(input_ids)
                    
                    # Calculate language modeling loss against expected output
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_targets = output_ids[:, 1:].contiguous() if output_ids.size(1) > 1 else input_ids[:, 1:].contiguous()
                    lm_loss = lm_criterion(shift_logits.view(-1, self.config.vocab_size), 
                                         shift_targets.view(-1))
                    
                    # No explicit bridge loss in this phase
                    bridge_loss = torch.tensor(0.0, device=self.device)
                    loss = lm_loss
                
                elif phase == 'rl':
                    # Simplified RL implementation (PPO-like)
                    # In a full implementation, this would be more complex
                    
                    # First get policy outputs without retrieval
                    logits_old, bridge_signals, bridge_queries = self.model(input_ids)
                    probs_old = F.softmax(logits_old, dim=-1)
                    
                    # If bridge activated, get external info and do second pass
                    if bridge_signals and any(b[1].any() for b in bridge_signals):
                        # Process each example in batch
                        external_infos = []
                        
                        for i, (activated, query, query_text) in enumerate(zip(
                            [any(b[1][i].item() for b in bridge_signals) for i in range(input_ids.size(0))],
                            bridge_queries[-1] if bridge_queries else [None] * input_ids.size(0),
                            batch.get('query_repr', [None] * input_ids.size(0))
                        )):
                            if activated:
                                # Get external information
                                external_info = self.ext_service.get_information(query, query_text=query_text)
                                external_infos.append(external_info)
                            else:
                                external_infos.append(None)
                        
                        # Forward pass with external info
                        logits_new, _, _ = self.model(input_ids, external_info=external_infos)
                        probs_new = F.softmax(logits_new, dim=-1)
                        
                        # Calculate output token accuracy against expected output
                        pred_new = torch.argmax(logits_new[:, -1, :], dim=-1)
                        pred_old = torch.argmax(logits_old[:, -1, :], dim=-1)
                        target = output_ids[:, 0] if output_ids.size(1) > 0 else input_ids[:, 0]
                        
                        # Calculate rewards based on accuracy improvement
                        reward = (pred_new == target).float() - (pred_old == target).float()
                        
                        # Calculate KL divergence between old and new policies
                        kl = torch.sum(probs_old * (torch.log(probs_old + 1e-10) - torch.log(probs_new + 1e-10)), dim=-1)
                        
                        # PPO-style clipped objective
                        ratio = probs_new / (probs_old + 1e-10)
                        surr1 = ratio * reward
                        surr2 = torch.clamp(ratio, 0.8, 1.2) * reward
                        actor_loss = -torch.min(surr1, surr2).mean()
                        
                        # Add KL penalty
                        loss = actor_loss + 0.01 * kl.mean()
                        bridge_loss = torch.tensor(0.0, device=self.device)
                        lm_loss = torch.tensor(0.0, device=self.device)
                    else:
                        # If no bridge, just use standard LM loss
                        shift_logits = logits_old[:, :-1, :].contiguous()
                        shift_targets = input_ids[:, 1:].contiguous()
                        lm_loss = lm_criterion(shift_logits.view(-1, self.config.vocab_size), shift_targets.view(-1))
                        bridge_loss = torch.tensor(0.0, device=self.device)
                        loss = lm_loss
                
                # Backward pass and optimization
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                
                # Record losses
                train_losses.append(loss.item())
                if phase != 'rl':  # RL phase doesn't have separate lm_loss and bridge_loss
                    train_lm_losses.append(lm_loss.item())
                    train_bridge_losses.append(bridge_loss.item())
                
                # Logging
                if step % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Step {step}, Loss: {loss.item():.4f}, "
                          f"LM Loss: {lm_loss.item() if lm_loss is not torch.tensor(0.0, device=self.device) else 0:.4f}, "
                          f"Bridge Loss: {bridge_loss.item() if bridge_loss is not torch.tensor(0.0, device=self.device) else 0:.4f}")
                    
                    # Update tensorboard
                    self.writer.add_scalar('train/loss', loss.item(), epoch * len(self.train_loader) + step)
                    self.writer.add_scalar('train/lm_loss', lm_loss.item() if lm_loss is not torch.tensor(0.0, device=self.device) else 0, 
                                          epoch * len(self.train_loader) + step)
                    self.writer.add_scalar('train/bridge_loss', bridge_loss.item() if bridge_loss is not torch.tensor(0.0, device=self.device) else 0, 
                                          epoch * len(self.train_loader) + step)
            
            # End of epoch
            train_loss = sum(train_losses) / len(train_losses)
            train_lm_loss = sum(train_lm_losses) / len(train_lm_losses) if train_lm_losses else 0
            train_bridge_loss = sum(train_bridge_losses) / len(train_bridge_losses) if train_bridge_losses else 0
            
            # Validation phase
            val_loss = self.validate(phase)
            
            # Step learning rate scheduler
            scheduler.step()
            
            # Log epoch stats
            print(f"Epoch {epoch+1}/{epochs} completed in {time.time()-start_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/train_lm_loss', train_lm_loss, epoch)
            self.writer.add_scalar('epoch/train_bridge_loss', train_bridge_loss, epoch)
            
            # Save checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'config': self.config
                }
                torch.save(checkpoint, os.path.join(self.save_dir, f'bridge_model_{phase}_best.pt'))
                print(f"Saved best model checkpoint with val_loss {val_loss:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'config': self.config
                }
                torch.save(checkpoint, os.path.join(self.save_dir, f'bridge_model_{phase}_epoch{epoch+1}.pt'))
    
    def validate(self, phase='supervised'):
        """Run validation and return average loss"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                bridge_labels = batch['bridge_label'].to(self.device)
                output_ids = batch['output_ids'].to(self.device)
                
                # Forward pass based on phase
                if phase in ['supervised', 'bridge_detection']:
                    logits, bridge_signals, _ = self.model(input_ids)
                    
                    # Language modeling loss
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_targets = input_ids[:, 1:].contiguous()
                    lm_loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), 
                                             shift_targets.view(-1))
                    
                    # Using combined loss for validation as well
                    val_losses.append(lm_loss.item())
                    
                elif phase in ['bridge_retrieval', 'rl']:
                    # For these phases, we validate on next token prediction accuracy
                    # Get query representation from batch if available
                    query_reprs = batch.get('query_repr', [None] * input_ids.size(0))
                    
                    # First pass to detect bridge activation
                    _, bridge_signals, bridge_queries = self.model(input_ids)
                    
                    # If bridge activated, get external info
                    if bridge_signals and any(b[1].any() for b in bridge_signals):
                        # Process each example in batch
                        external_infos = []
                        
                        for i, (activated, query) in enumerate(zip(
                            [any(b[1][i].item() for b in bridge_signals) for i in range(input_ids.size(0))],
                            bridge_queries[-1] if bridge_queries else [None] * input_ids.size(0)
                        )):
                            if activated:
                                # Use query representation from data if available
                                query_text = query_reprs[i] if i < len(query_reprs) else None
                                external_info = self.ext_service.get_information(query, query_text=query_text)
                                external_infos.append(external_info)
                            else:
                                external_infos.append(None)
                        
                        logits, _, _ = self.model(input_ids, external_info=external_infos)
                    else:
                        logits, _, _ = self.model(input_ids)
                    
                    # Use output_ids as targets if available
                    if output_ids.size(1) > 1:
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_targets = output_ids[:, 1:].contiguous()
                    else:
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_targets = input_ids[:, 1:].contiguous()
                    
                    val_loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), 
                                              shift_targets.view(-1))
                    val_losses.append(val_loss.item())
        
        return sum(val_losses) / len(val_losses)
    
    def generate_sample(self, prompt, max_tokens=100, temperature=0.8):
        """Generate sample text with the model"""
        self.model.eval()
        
        # Tokenize prompt
        input_ids = torch.tensor([self.model.tokenizer.encode(prompt)], device=self.device)
        
        # Generate
        generated_ids = self.model.generate(
            input_ids, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            external_service=self.ext_service
        )
        
        # Decode
        generated_text = self.model.tokenizer.decode(generated_ids[0].tolist())
        
        return generated_text


# -----------------------------------------------------------------------------
# Main Training Script
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Bridge Model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to existing dataset')
    parser.add_argument('--generate_data', action='store_true',
                        help='Generate new dataset')
    parser.add_argument('--data_size', type=int, default=5000,
                        help='Size of generated dataset')
    parser.add_argument('--lm_studio_endpoint', type=str, default='http://localhost:1234/v1',
                        help='Endpoint for LM Studio API (used for data generation)')
    
    # Knowledge service arguments
    parser.add_argument('--knowledge_db', type=str, default='knowledge.db',
                        help='Path to knowledge database')
    parser.add_argument('--search_api_key', type=str, default=None,
                        help='API key for web search')
    
    # Model arguments
    parser.add_argument('--n_layer', type=int, default=12,
                        help='Number of layers')
    parser.add_argument('--n_head', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--bridge_neurons_pct', type=float, default=0.05,
                        help='Percentage of neurons dedicated as bridge')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--phase', type=str, default='supervised',
                        choices=['supervised', 'bridge_detection', 'bridge_retrieval', 'rl'],
                        help='Training phase')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Number of warmup epochs')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--output_dataset', type=str, default='bridge_dataset.json',
                        help='Path to save generated dataset')
    
    args = parser.parse_args()
    
    # Create save and log directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Data preparation
    if args.generate_data or not args.data_path or not os.path.exists(args.data_path):
        print(f"Generating new dataset of size {args.data_size}...")
        # Use LLMDataGenerator for higher quality data
        generator = LLMDataGenerator(lm_studio_endpoint=args.lm_studio_endpoint)
        data = generator.generate_dataset(size=args.data_size)
        generator.save_dataset(data, args.output_dataset)
        data_path = args.output_dataset
    else:
        print(f"Loading dataset from {args.data_path}...")
        data_path = args.data_path
        # Load dataset
        generator = LLMDataGenerator()
        data = generator.load_dataset(data_path)
    
    # Initialize tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create dataset and dataloaders
    dataset = BridgeDataset(data, tokenizer)
    train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size=args.batch_size)
    print(f"Created dataloaders with {len(train_loader)} training batches")
    
    # Initialize external knowledge service if database provided
    if os.path.exists(args.knowledge_db):
        external_service = RealExternalKnowledgeService(
            vector_db_path=args.knowledge_db,
            search_api_key=args.search_api_key
        )
        print(f"Initialized external knowledge service with database: {args.knowledge_db}")
    else:
        external_service = None
        print("No knowledge database found, using mock service")
    
    # Create model configuration
    config = BridgeModelConfig()
    config.n_layer = args.n_layer
    config.n_head = args.n_head
    config.n_embd = args.n_embd
    config.dropout = args.dropout
    config.bridge_neurons_pct = args.bridge_neurons_pct
    
    # Create or load model
    if args.load_checkpoint:
        print(f"Loading model from checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model = BridgeTransformer(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Creating new model")
        model = BridgeTransformer(config).to(device)
    
    # Calculate model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count / 1e6:.2f}M parameters")
    
    # Create trainer
    trainer = BridgeTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        external_service=external_service
    )
    
    # Train model
    print(f"Starting training in {args.phase} phase for {args.epochs} epochs...")
    trainer.train(
        epochs=args.epochs, 
        lr=args.lr, 
        warmup_epochs=args.warmup_epochs,
        phase=args.phase
    )
    
    # Generate some samples
    print("\nGenerating samples...")
    samples = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "Hello, how are you today?",
        "When was Einstein born?"
    ]
    
    for sample in samples:
        print(f"\nPrompt: {sample}")
        generated = trainer.generate_sample(sample, max_tokens=50)
        print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
