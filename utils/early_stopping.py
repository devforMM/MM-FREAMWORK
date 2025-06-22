import torch

def early_stopping(train_fn, val_fn, max_epochs=100, patience=3):
    losses = []
    val_losses = []
    best_loss = float('inf')
    counter = 0

    for epoch in range(max_epochs):
        # Training step (à remplacer par ta fonction ou ton code de training)
        loss = train_fn()
        losses.append(loss)

        # Validation step (à remplacer par ta fonction ou ton code de validation)
        val_loss = val_fn()
        val_losses.append(val_loss)

        # Check for improvement
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0  # Reset patience counter
        else:
            counter += 1

        # Early stopping condition
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return losses, val_losses
