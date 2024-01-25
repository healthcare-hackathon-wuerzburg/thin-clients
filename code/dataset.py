from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self):
        # we need path for the folder of the images, as well as the path to the .csv file
        # we map each image (based on name) to the vector inside the .csv file
        # image has to be transformed to a tensor
        pass

    def __len__(self):
        # idea: our Dataset will be a list of tuples of (tensor, vector), list lenght should be fine
        pass

    def __getitem__(self, item):
        # idea: get pair of image(tensor) and vector (for classification)
        pass
