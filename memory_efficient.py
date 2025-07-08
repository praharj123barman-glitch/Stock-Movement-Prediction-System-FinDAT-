# memory_efficient.py
import gc
import torch

def memory_cleanup():
    """Clean up memory to prevent OOM errors"""
    gc.collect()
    torch.cuda.empty_cache()

def gradient_accumulation_training(model, train_loader, optimizer, accumulation_steps=4, device="cpu"):
    """Train model with gradient accumulation to save memory"""
    model.train()
    total_loss = 0.0
    steps = 0
    
    # Set gradients to zero initially
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_loader):
        # Get batch data
        stock_features, adj_matrix, sector_indices, return_labels, movement_labels = batch
        
        # Move to device
        stock_features = stock_features.to(device)
        adj_matrix = adj_matrix.to(device)
        sector_indices = sector_indices.to(device)
        return_labels = return_labels.to(device)
        movement_labels = movement_labels.to(device)
        
        # Forward pass
        return_preds, movement_preds = model(stock_features, adj_matrix, sector_indices)
        
        # Calculate loss and normalize by accumulation steps
        loss = model.calculate_loss(
            return_preds, movement_preds, return_labels, movement_labels
        ) / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update total loss
        total_loss += loss.item() * accumulation_steps
        steps += 1
        
        # Step optimizer after accumulation_steps or at the end of epoch
        if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            
            # Free up memory
            memory_cleanup()
    
    return total_loss / steps
