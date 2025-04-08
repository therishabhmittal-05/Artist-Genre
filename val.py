import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def validate_one_epoch(model, val_loader, loss_artist_fn, loss_genre_fn, device, epoch):
    model.eval()
    all_artist_preds = []
    all_artist_true = []
    all_genre_preds = []
    all_genre_true = []
    
    val_loss = 0.0
    with torch.no_grad():
        for inputs, val_artist, val_genre in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            inputs = inputs.to(device)
            val_artist = val_artist.to(device)
            val_genre = val_genre.to(device).float()

            artist_op, genre_op = model(inputs)

            artist_l = loss_artist_fn(artist_op, val_artist)
            genre_l = loss_genre_fn(genre_op, val_genre)
            loss = artist_l + genre_l
            val_loss += loss.item()

            artist_hat = torch.argmax(artist_op, dim=1)
            all_artist_preds.extend(artist_hat.detach().cpu().numpy())
            all_artist_true.extend(val_artist.detach().cpu().numpy())

            genre_hat = torch.sigmoid(genre_op)
            genre_hat = (genre_hat > 0.5).int()
            all_genre_preds.extend(genre_hat.detach().cpu().numpy())
            all_genre_true.extend(val_genre.detach().cpu().numpy())

    artist_acc = accuracy_score(all_artist_true, all_artist_preds)
    genre_f1 = f1_score(all_genre_true, all_genre_preds, average='micro')

    avg_val_loss = val_loss / len(val_loader)

    return avg_val_loss, artist_acc, genre_f1
