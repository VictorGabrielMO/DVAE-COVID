import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

def create_sliding_windows(series, seq_len):
    """
    Args:
        series (1D array-like): The full time series, e.g., list or np.ndarray of Ct values
        seq_len (int): Length of each sliding window
    Returns:
        windows: np.ndarray of shape (N_samples, seq_len, 1)
    """
    #series = np.array(series, dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(series, seq_len)
    windows = windows.reshape(-1, seq_len, 1)  # add x_dim = 1
    return windows

def random_split_windows(windows, val_ratio, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(windows))
    val_size = int(len(windows) * val_ratio)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return windows[train_idx], windows[val_idx]


def build_dataloader(cfg):
    """
    Splits the time series into training and validation datasets, builds loaders.
    Returns batches of shape: (seq_len, batch_size, 1)
    """
    data_dir = cfg.get('User', 'data_dir')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    sequence_len = cfg.getint('DataFrame', 'sequence_len')
    val_ratio = cfg.getfloat('DataFrame', 'val_ratio')
    
    ct_values_array = np.load(data_dir)
    
    # Build (N_samples, seq_len, 1) sliding windows
    sliding_windows = create_sliding_windows(ct_values_array, sequence_len)
    print(sliding_windows.shape)
    
    # Split into train and validation
    train_windows, val_windows = random_split_windows(sliding_windows, val_ratio=val_ratio)

    train_dataset = CtSeriesDataset(train_windows)
    val_dataset = CtSeriesDataset(val_windows)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, len(train_loader), len(val_loader)

    

class CtSeriesDataset(Dataset):
    def __init__(self, windows):
        """
        Args:
            ct_values (np.ndarray or list or pd.Series): A 1D array of Ct values (time-ordered).
            sequence_len (int): Number of time steps per sequence.
        """
        super().__init__()
        
        # get ct values array
        self.windows = torch.tensor(windows, dtype=torch.float32)
        
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]

if __name__ == '__main__':
    train_loader, val_loader = build_dataloader(data_dir="./data/X_COVID.npy",
                                                sequence_len=5,
                                                batch_size=32)
    for seq in val_loader:
        print(seq.shape)

