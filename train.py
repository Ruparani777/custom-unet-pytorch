import torch
from model import UNet
from torch.utils.data import DataLoader
from dataset import CustomDataset

def train():
    model = UNet()
    dataset = CustomDataset('data/')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(3):
        for images, masks in dataloader:
            preds = model(images)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()}")

if __name__ == "__main__":
    train()
