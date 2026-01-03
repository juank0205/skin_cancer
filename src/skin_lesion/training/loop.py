import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    epochs: int = 20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    for epoch in range(epochs):
        # ======================
        # TRAIN
        # ======================
        model.train()
        train_loss = 0.0
        train_count = 0

        for X, y in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [train]",
            leave=False,
        ):
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_count += X.size(0)

        train_loss /= train_count

        # ======================
        # VALIDATION
        # ======================
        model.eval()
        val_loss = 0.0
        val_count = 0
        correct = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                logits = model(X)
                loss = criterion(logits, y)

                val_loss += loss.item() * X.size(0)
                val_count += X.size(0)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()

        val_loss /= val_count
        val_acc = correct / val_count

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )
