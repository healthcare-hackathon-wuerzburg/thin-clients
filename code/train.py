import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CustomDataset
from model import SimpleModel


def main() -> None:
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    batch_size = 1
    epochs = 100
    num_workers = 4

    # Model details
    model = SimpleModel().to(device)
    criterion = nn.BCELoss()  # maybe some other loss for comparing percentages
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loading and preparing data
    images_folder = 'data/images/train/detail'
    csv_file = 'data/labels.csv'
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(images_folder, csv_file, transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Training Loop
    for epoch in tqdm(range(epochs), desc='Training', dynamic_ncols=True):
        total_loss = 0.0
        for image, target in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', dynamic_ncols=True):
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), 'trained_models/simple_model.pth')


if __name__ == '__main__':
    main()
