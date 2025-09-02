import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import DataModule
from model import ColaModel
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import wandb
import hydra

class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()
        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        val_batch = {k: v.to(pl_module.device) if torch.is_tensor(v) else v
                     for k, v in val_batch.items()}

        outputs = pl_module(
            input_ids=val_batch["input_ids"],
            attention_mask=val_batch["attention_mask"]
        )
        preds = torch.argmax(outputs.logits, dim=1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {
                "Label": labels.cpu().numpy(),
                "Predicted": preds.cpu().numpy()
            }
        )

        wrong_df = df[df["Label"] != df["Predicted"]]

        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


# NOTE: Need to provide the path for configs folder and the config file name
# ref this tutorial to know more hydra: https://www.sscardapane.it/tutorials/hydra-tutorial/#executing-multiple-runs
@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )

    cola_model = ColaModel(cfg.model.name)

    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        filename="best-checkpoint.ckpt",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    # Configure device for training
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    wandb_logger = WandbLogger(project="MLOps Basics")

    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator=accelerator,
        logger=wandb_logger,
        devices=devices,
        log_every_n_steps=cfg.training.log_every_n_steps,
        max_epochs=cfg.training.max_epochs,
        fast_dev_run=False,
        # logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
        deterministic=cfg.training.deterministic,
        callbacks=[checkpoint_callback,SamplesVisualisationLogger(cola_data), early_stopping_callback],
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches
    )
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()
