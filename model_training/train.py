import hydra
import pandas as pd
import pytorch_lightning as pl
import torch 
import wandb

from omegaconf.omegaconf import OmegaConf
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import HabermanDataModule
from ml_model import HabermanANN

config_dir = Path.cwd().joinpath("configs")
data_path = Path.cwd().joinpath("data", "haberman.csv")

class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()
        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        # can be done on complete dataset also
        val_batch = next(iter(self.datamodule.val_dataloader()))
        input = val_batch["features"]

        # get the predictions
        outputs = pl_module(input)
        preds = torch.argmax(outputs, 1)
        labels = val_batch["targets"]

        # predicted and labelled data
        df = pd.DataFrame(data=input, columns=['age', 'year', 'n_aux_nodes'])
        df['Label'] = labels.numpy()
        df['Predicted'] = preds.numpy()

        # wrongly predicted data
        wrong_df = df[df["Label"] != df["Predicted"]]

        # Logging wrongly predicted dataframe as a table
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


@hydra.main(config_path=config_dir, config_name="config")
def train(cfg):
    train_data = HabermanDataModule(data_path, 
        cfg.training.train_bs, 
        cfg.training.val_bs
    )
    print("Got the data bitches!")


if __name__ == "__main__":
    train()

