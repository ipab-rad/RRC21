from torch.utils.data import  Dataset

class GraspMetricDataset(Dataset):
    """
    TODO: finish dataset loader 
    Dataset for Grasp Metric Learning
    conceptualized as binary matrix of with:
    - axis 0: object position
    - axis 1: grasp configuration
    - value: 1 if grasp can be succesful to bring to position, and 0 otherwise
    """

    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return idx