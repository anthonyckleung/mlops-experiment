import pandas as pd
import pytorch_lightning as pl
import torch

from torch.utils.data import Dataset, DataLoader, random_split


class HabermanData(Dataset):
    def __init__(self, data_df):
        data_df = data_df.reset_index(drop=True)
        self.features = data_df[['age','year', 'n_auxillary_nodes']].values
        self.target = data_df['status'].values
        
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float)
        targets = torch.tensor(self.target[idx], dtype = torch.long)

        return {"features": features,
                "targets": targets}



class HabermanDataModule(pl.LightningDataModule):
    def __init__(self, data_path, train_bs, val_bs):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.train_bs = train_bs 
        self.val_bs = val_bs
        
    def setup(self, stage=None):
        data = self.data.sample(frac=1).reset_index(drop=True)
        data['status'] = data['status'] - 1

        dataset = HabermanData(data)

        # Split data into train, validation, and testing
        trainval_size = int(0.8 * len(data))
        train_size = int(trainval_size*0.8)

        val_size = trainval_size - train_size
        test_size = len(data) - trainval_size

        trainval_set, test_set = random_split(dataset, [trainval_size, test_size])
        train_set, val_set = random_split(trainval_set, [train_size, val_size])
        
        if (stage == 'fit') or (stage is None):
            self.training_set = train_set
            self.validation_set = val_set
        
        if stage == 'test':
            self.testing_set = test_set
 
    def train_dataloader(self):
        train_loader = DataLoader(
            self.training_set,
            batch_size = self.train_bs,
            shuffle=False,
            num_workers = 0
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.validation_set,
            batch_size = self.val_bs,
            shuffle=False,
            num_workers = 0
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.testing_set,
            batch_size = self.val_bs,
            shuffle=False,
            num_workers = 0
        )
        return test_loader