import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

    # initialize empty result arrays
    all_targets = []
    all_outputs = []

    for image, target in tqdm(evaluate_loader, dynamic_ncols=True, desc="Evaluate"):
        image, target = image.to(device), target.to(device)
        with torch.no_grad():
            output = model(image)

        # round output to integer values
        output_rounded = round_with_threshold(output, 0.2)

        # convert to numpy to be able to use sklearn performance metrics
        target_np = target.int().cpu().view(-1).numpy()
        output_rounded_np = output_rounded.cpu().view(-1).numpy()

        # accumulate targets and outputs
        all_targets.append(target_np)
        all_outputs.append(output_rounded_np)

    # define averaging method for precision recall and f1 scores
    avg_method = "weighted"

    # calculate sklearn performance metrics
    accuracy = accuracy_score(all_targets, all_outputs)
    precision = precision_score(all_targets, all_outputs, average=avg_method)
    recall = recall_score(all_targets, all_outputs, average=avg_method)
    f1 = f1_score(all_targets, all_outputs, average=avg_method)

    # Print results
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


if __name__ == '__main__':
    main()
