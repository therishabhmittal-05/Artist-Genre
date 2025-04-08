import torch
import torch.nn as nn
from train import train_one_epoch
from val import validate_one_epoch
from data.data_preprocessing import check_image_files

img_root = "/kaggle/input/wikiart"
df = pd.read_csv("your_file.csv")

# Clean the data
df = check_image_files(df, img_root)


loss_artist = nn.CrossEntropyLoss()
loss_genre = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)




for epoch in range(n_epochs):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, device,
        loss_artist, loss_genre, epoch
    )

    val_loss, artist_acc, genre_f1 = validate_one_epoch(
        model, val_loader, device,
        loss_artist, loss_genre, epoch
    )

    print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Artist Acc: {artist_acc:.4f} | Genre F1: {genre_f1:.4f}")

    scheduler.step(val_loss)
    es(val_loss)
    if es.early_stop:
        print("Early stopping triggered.")
        break
