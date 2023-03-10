from torch.utils.data import Dataset
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split

class DataStorage(Dataset):
    def __init__(self, path):
        """
        Initialization of DataStorage object
        parameters:
            path (Path or list): If path, then it denotes the folder to read. If a list, it denotes the list of graphs which self.graphs should be.
        return: None
        """
        if isinstance(path, Path):
            super().__init__()
            graphs = list(path.glob("*.pt"))
            train_data, test_data = train_test_split(graphs, test_size=0.2, random_state=17)
            self.train_graphs = train_data
            self.test_graphs = test_data
        else: 
            self.graphs = path
    
    def append(self, path: Path):
        """
        Add more graphs to self.graph given a file name
        parameters:
            path (Path): Denotes the folder to open and parse for graphs
        return: None
        """
        graphs = list(path.glob("*.pt"))
        train_data, test_data = train_test_split(graphs, test_size=0.2, random_state=17)
        self.train_graphs = self.train_graphs + train_data
        self.test_graphs = self.test_graphs + test_data

    def __getitem__(self, idx):
        """
        Get item in the training set at given index
        parameters:
            idx (int): Index to search within train_graphs
        return: Graph corresponding to given index
        """
        return torch.load(self.train_graphs[idx])
    
    def test_get(self, idx):
        """
        Get item in the test set at given index
        parameters:
            idx (int): Index to search within test_graphs
        return: Graph corresponding to given index
        """
        return torch.load(self.test_graphs[idx])
    
    def __len__(self) -> int:
        """
        Get item in the test set at given index
        parameters:
            None
        return: The number of training graphs
        """
        return len(self.train_graphs)
    
    def test_len(self):
        """
        Get item in the test set at given index
        parameters:
            None
        return: The number of test graphs
        """
        return len(self.test_graphs)