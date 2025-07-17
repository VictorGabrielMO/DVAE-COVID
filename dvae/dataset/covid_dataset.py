import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import pandas as pd

def build_dataloader(cfg):
    """
    Splits the time series into training and validation datasets, builds loaders.
    Returns batches of shape: (seq_len, batch_size, features)
    """
    # Load hyperparameters
    input_col_list = ["Ct_Value"]
    target_col_list = ["Ct_Value"]
    df_path = cfg.get('User', 'data_dir')
    window_length = cfg.getint('DataFrame', 'window_length')
    step_length = cfg.getint('DataFrame', 'step_length')
    context_length = cfg.getint('DataFrame', 'context_length')
    validation_ratio = cfg.getfloat('DataFrame', 'validation_ratio')
    random_split_flag = cfg.getboolean('DataFrame', 'random_split_flag')
    seed = cfg.getint('DataFrame', 'seed')
    batch_size = cfg.getint('DataFrame', 'batch_size')

    # Preprocess 
    all_windows = data_preprocessing(
        df_path=df_path,
        input_col_list=input_col_list,
        target_col_list=target_col_list,
        window_length=window_length,
        step_length=step_length,
        seed=seed,
        zero_anchor_flag=True,
    )

    # Split data in train and validation
    train_windows, valid_windows = train_validation_split(
        all_windows, validation_ratio, random_split_flag,
        window_length, step_length, seed
    )

    # Create datasets
    train_dataset = HaddadDataset(train_windows, input_col_list, context_length)
    valid_dataset = HaddadDataset(valid_windows, input_col_list, context_length)

    # Create loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    return train_dataloader, valid_dataloader, len(train_dataset), len(valid_dataset)


def data_preprocessing(df_path, input_col_list, target_col_list,
                       window_length, step_length, seed, zero_anchor_flag=True):
    df = pd.read_csv(df_path)
    df["Date"] = pd.to_datetime(df['Date'])

    col_list = input_col_list + target_col_list
    new_col_list = [f"{col}_filled" for col in col_list]

    if zero_anchor_flag:
        def zero_anchor_and_interpolate(group):
            result = pd.DataFrame(index=group.index)
            for col in col_list:
                s = group[col].copy()
                s.iloc[0] = 0 if pd.isna(s.iloc[0]) else s.iloc[0]
                s.iloc[-1] = 0 if pd.isna(s.iloc[-1]) else s.iloc[-1]
                s = s.interpolate(method="linear")
                result[f"{col}_filled"] = s
            return result

        filled_df = df.drop(columns="City").groupby(df["City"], group_keys=False).apply(zero_anchor_and_interpolate)
        df = df.assign(**filled_df)
    else:
        for col in col_list:
            new_col = f"{col}_filled"
            df[new_col] = df.groupby("City", group_keys=False)[col].transform(
                lambda g: g.interpolate(method="linear", limit_area="inside").fillna(0)
            )

    window_list = []
    for _, group_df in df.groupby("City"):
        city_data = torch.tensor(group_df[new_col_list].values, dtype=torch.float32)  # [T, F]
        if city_data.size(0) < window_length:
            continue
        windows = city_data.unfold(dimension=0, size=window_length, step=step_length)
        windows = windows.contiguous().view(-1, window_length, len(new_col_list))
        window_list.append(windows)

    window_list = torch.cat(window_list, dim=0)
    return window_list


def train_validation_split(window_list, validation_ratio,
                           random_split_flag, window_length, step_length, seed):
    valid_size = int(validation_ratio * len(window_list))
    train_size = len(window_list) - valid_size

    if random_split_flag:
        return random_split_without_overlap(window_list, valid_size, window_length, step_length, seed)
    else:
        return window_list[:train_size], window_list[train_size:]


def random_split_without_overlap(window_list, val_size, window_length=24, step_length=12, seed=48):
    if step_length >= window_length:
        return random_split(window_list, [len(window_list) - val_size, val_size], generator=torch.Generator().manual_seed(seed))

    random.seed(seed)
    train_positions = list(range(len(window_list)))
    val_positions = []

    while len(val_positions) < val_size:
        idx = random.randint(0, len(train_positions) - 1)
        val_positions.append(train_positions[idx])

        # Prune left
        left_i = idx - 1
        while left_i >= 0:
            start_left = train_positions[left_i] * step_length
            if start_left + window_length <= train_positions[idx] * step_length:
                break
            train_positions.pop(left_i)
            idx -= 1
            left_i -= 1

        # Prune right
        while idx < len(train_positions) - 1:
            start_right = train_positions[idx + 1] * step_length
            if start_right >= train_positions[idx] * step_length + window_length:
                break
            train_positions.pop(idx + 1)

        train_positions.pop(idx)

    train_subset = [window_list[i] for i in train_positions]
    val_subset = [window_list[i] for i in val_positions]

    return train_subset, val_subset


class HaddadDataset(Dataset):
    def __init__(self, windows, input_col_list, context_length=12, mask_value=-1):
        super().__init__()
        self.num_input_feature = len(input_col_list)
        self.context_length = context_length
        self.mask_value = mask_value
        self.data = windows

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        full_window = self.data[index]
        
        # Takes only the input features
        input_window = full_window[:, :self.num_input_feature]
        
        # Masks the input
        input_window[self.context_length:] = self.mask_value
        
        # Takes the target features
        target_window = full_window[:, self.num_input_feature:]
        
        return input_window, target_window
