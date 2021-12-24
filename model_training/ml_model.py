import hydra
import pytorch_lightning as pl
import torch 
import torch.nn as nn
import torchmetrics
import wandb

from omegaconf import OmegaConf
from torch.nn import functional as F
from sklearn.metrics import accuracy_score


class HabermanANN(pl.LightningModule):
    def __init__(self, lr=1e-2):
        super(HabermanANN, self).__init__()
        self.num_classes = 2
        self.save_hyperparameters()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 2)

        self.ann = nn.Sequential(
                nn.Linear(3, 10),
                nn.ReLU(),
                nn.Linear(10, 2),
        )
        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")
        

    def forward(self, x):
        x = self.ann(x)
        return x


    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["features"])
        loss = self.loss_func(logits, batch['targets'])
        train_acc = self.train_accuracy_metric(logits, batch["targets"])
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["features"])
        loss = self.loss_func(logits, batch['targets'])
        _, preds = torch.max(logits, dim=1)
        val_acc = accuracy_score(preds.cpu(), batch["targets"].cpu())
        val_acc = torch.tensor(val_acc)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, batch["targets"])
        precision_macro = self.precision_macro_metric(preds, batch["targets"])
        recall_macro = self.recall_macro_metric(preds, batch["targets"])
        precision_micro = self.precision_micro_metric(preds, batch["targets"])
        recall_micro = self.recall_micro_metric(preds, batch["targets"])
        f1 = self.f1_metric(preds, batch["targets"])

        self.log("val/loss", loss, prog_bar=True, on_step=True)
        self.log("val/acc", val_acc, prog_bar=True)
        self.log("val/precision_macro", precision_macro, prog_bar=True)
        self.log("val/recall_macro",recall_macro, prog_bar=True)
        self.log("val/precision_micro", precision_micro, prog_bar=True)
        self.log("val/recall_micro", recall_micro, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)
        return {"targets": batch["targets"], "logits": logits}

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch["features"])
        loss = self.loss_func(logits, batch['targets'])
        _, preds = torch.max(logits, dim=1)
        test_acc = accuracy_score(preds.cpu(), batch["targets"].cpu())
        self.log("test_loss", loss)
        self.log("test_acc", test_acc) 
        

    def loss_func(self, pred, target):
        return F.cross_entropy(pred, target)

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x["targets"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)

        ## There are multiple ways to track the metrics
        # 1. Confusion matrix plotting using inbuilt W&B method
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.numpy(), y_true=labels.numpy()
                )
            }
        )

        # 2. Confusion Matrix plotting using scikit-learn method
        # wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds)})

        # 3. Confusion Matric plotting using Seaborn
        # data = confusion_matrix(labels.numpy(), preds.numpy())
        # df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        # df_cm.index.name = "Actual"
        # df_cm.columns.name = "Predicted"
        # plt.figure(figsize=(7, 4))
        # plot = sns.heatmap(
        #     df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}
        # )  # font size
        # self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

        # self.logger.experiment.log(
        #     {"roc": wandb.plot.roc_curve(labels.numpy(), logits.numpy())}
        # )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        