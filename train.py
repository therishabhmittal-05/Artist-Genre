import torch
from tqdm import tqdm

def train_one_epoch(model, train_loader, optimizer, device, loss_artist_fn, loss_genre_fn, epoch):
    model.train()
    running_loss = 0.0

    for inputs, artist_label, genre_label in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
        inputs = inputs.to(device)
        artist_label = artist_label.to(device)
        genre_label = genre_label.to(device).float()

        optimizer.zero_grad()
        artist_op, genre_op = model(inputs)

        artist_loss = loss_artist_fn(artist_op, artist_label)
        genre_loss = loss_genre_fn(genre_op, genre_label)
        loss = artist_loss + genre_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss
