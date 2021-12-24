import hydra
import os
import pandas as pd
import pytorch_lightning as pl
import torch 
import wandb

from dotenv import load_dotenv
from omegaconf.omegaconf import OmegaConf
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import HabermanDataModule
from ml_model import HabermanANN

load_dotenv()

# Login to Weights & Biases
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)

# Get all relevent configs from Hydra
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
    root_dir = Path(hydra.utils.get_original_cwd())
    haberman_data = HabermanDataModule(data_path, 
        cfg.training.train_bs, 
        cfg.training.val_bs
    )
    haberman_model = HabermanANN()

    checkpoint_callback = ModelCheckpoint(
        dirpath = Path.joinpath(root_dir, "models"),
        filename="best-checkpoint",
        monitor = "val/loss",
        mode = "min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/loss", patience=3, verbose=True, mode="min"
    )

    
    wandb_logger = WandbLogger(project="MLOps Basics")

    trainer = pl.Trainer(
        # gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=10,
        fast_dev_run=False,
        accelerator='cpu',
        log_every_n_steps = 5,
        logger=wandb_logger,
        callbacks = [checkpoint_callback, 
        SamplesVisualisationLogger(haberman_data),
        early_stopping_callback]
    )
    trainer.fit(haberman_model, haberman_data)

    # Perform evaluation
    trainer.test(haberman_model, haberman_data)
    wandb.finish()


if __name__=="__main__":
    train()

