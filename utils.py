from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import WikiArtDs

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

def get_data_loaders(df, img_dir, batch_size=128):
    train_df = df[df.subset == 'train']
    val_df = df[df.subset == 'test']

    transform = get_transforms()

    train_dataset = WikiArtDs(train_df, img_dir, transform)
    val_dataset = WikiArtDs(val_df, img_dir, transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True
    )

    return train_loader, val_loader

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True