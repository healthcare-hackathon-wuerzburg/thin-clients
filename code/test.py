import torch
from torchvision import transforms
from PIL import Image
from model import SimpleModel


def test(path_to_image: str) -> list[float]:
    """
    Test the SimpleModel on the given image.

    Parameters:
        path_to_image (str): Path to the image file for testing.

    Returns:
        list[float]: List containing the output vector values.
    """
    # Process input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    image = Image.open(path_to_image)

    if transform:
        image = transform(image)
    image = image.to(device)

    # Load model
    model_path = "trained_models/simple_model.pth"
    model = SimpleModel().to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Forward pass model to generate output
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        output = output.numpy().astype(float)
        output = output.reshape(-1).tolist()

    results = [output[0], output[1], output[2], output[3]]
    return results


def main() -> None:
    """
    Main function to test the SimpleModel on a sample image and print the results.
    """
    print(test("./data/images/test/blutung1.jpg"))


if __name__ == '__main__':
    main()