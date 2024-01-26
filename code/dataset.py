import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, images_folder_path, csv_file_path, transform=None):
        """
        Initialize the CustomDataset.

        Parameters:
            images_folder_path (str): Path to the folder containing images.
            csv_file_path (str): Path to the CSV file containing image-vector mapping.
            transform (torchvision.transforms.Compose, optional): Transformations to be applied to the image.
        """
        self.images_folder_path = images_folder_path
        self.transform = transform

        # Read CSV file into a pandas DataFrame
        self.df = pd.read_csv(csv_file_path)

        # Get the list of image names
        all_files = os.listdir(images_folder_path)
        image_names = [file for file in all_files if file.endswith(".jpg")]
        # Remove the ".jpg" extension
        image_names = [name[:-4] for name in image_names]
        self.image_names = image_names

        # Initialize a list of tuples containing images as tensors and vectors
        self.data = self.initialize_data()

    def initialize_data(self):
        """
        Initialize a list of tuples containing images as tensors and vectors.

        Returns:
            list: List of tuples containing (image as tensor, vector).
        """
        data = []

        for image_name in self.image_names:
            # Construct the path to the image
            image_path = os.path.join(self.images_folder_path, image_name + ".jpg")

            # Load the image using PIL
            image = Image.open(image_path)

            # Apply transformations if specified
            if self.transform:
                image = self.transform(image)

            # Check if image_name exists in the DataFrame
            assert image_name in self.df['ID'].values, f"Image name {image_name} not found in the DataFrame."

            # Get the vector corresponding to the image from the DataFrame
            vector = torch.tensor(self.df.loc[self.df['ID'] == image_name].drop('ID', axis=1).values[0], dtype=torch.float32)
            data.append((image, vector))

        return data

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the item (image, vector) at the specified index.

        Parameters:
            index (int): Index to retrieve the item.

        Returns:
            tuple: A tuple containing the image (as a tensor) and the corresponding vector.
        """
        return self.data[index]


def main() -> None:
    # Example usage:
    # Specify the paths and any desired transformations
    images_folder = 'data/images/train'
    csv_file = 'data/labels.csv'
    transform = transforms.Compose([transforms.ToTensor()])  # Example transformation (convert image to tensor)

    # Create an instance of CustomDataset
    custom_dataset = CustomDataset(images_folder, csv_file, transform)

    train_loader = DataLoader(dataset=custom_dataset, batch_size=1, shuffle=True, num_workers=4)

    # Accessing individual samples
    sample_image, sample_vector = custom_dataset[0]
    print("Sample Image:", sample_image)
    print("Sample Vector:", sample_vector)

    for tensor, vector in train_loader:
        print(f"Tensor {tensor} and Vector {vector}.")


if __name__ == '__main__':
    main()
