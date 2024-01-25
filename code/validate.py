import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CustomDataset
from model import SimpleModel

def round_with_threshold(tensor, threshold):
    return (tensor >= threshold).int()

def main() -> None:
    # Loading and preparing data
    images_folder = 'data/images/validation'
    csv_file = 'data/labels.csv'
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    evaluate_dataset = CustomDataset(images_folder, csv_file, transform)
    evaluate_loader = DataLoader(dataset=evaluate_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Loading model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "trained_models/simple_model.pth"
    model = SimpleModel().to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    threshold = 0.5
    for image, target in tqdm(evaluate_loader, dynamic_ncols=True, desc="Evaluate"):
        image, target = image.to(device), target.to(device)
        with torch.no_grad():
            output = model(image)
        print(target)
        print(round_with_threshold(output, threshold))
        break
        # calc performance instead of loss
        # loss = criterion(output, target)

# average_loss = total_loss / len(train_loader)


if __name__ == '__main__':
    main()
