from typing import Any, Dict, List, Sequence, Tuple, Union

import hydra
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from torch.optim import Optimizer

import numpy as np
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients, GuidedGradCam
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from src.common.utils import iterate_elements_in_batches, render_images

from src.pl_modules import resnets
from src.pl_modules import losses

# from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay


class MyModel(pl.LightningModule):
    def __init__(
            self,
            cfg: DictConfig,
            name,
            num_classes,
            final_nl,
            loss,
            self_supervised=False,
            num_samples=False,
            batch_size=False,
            task="binary",
            *args,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.name = name
        self.self_supervised = self_supervised
        # self.automatic_optimization = False
        self.num_classes = num_classes
        self.loss = getattr(losses, loss)  # Add this to the config
        if final_nl:
            self.final_nl = getattr(F, final_nl)
        else:
            self.final_nl = lambda x, dim: x
        if self.name == "resnet18":
            self.net = resnets.resnet18(pretrained=True, num_classes=num_classes)
        elif self.name == "simclr_resnet18":
            self.net = resnets.simclr_resnet18(
                pretrained=False,
                num_classes=num_classes,
                num_samples=num_samples,
                batch_size=batch_size)
        elif self.name == "simclr_resnet18_transfer":
            self.net = resnets.simclr_resnet18_transfer(
                pretrained=False,
                num_classes=num_classes,
                num_samples=num_samples,
                batch_size=batch_size)
        else:
            raise NotImplementedError("Could not find network {}.".format(self.net))

        metric = torchmetrics.Accuracy(task)
        self.train_accuracy = metric.clone().cuda()
        self.val_accuracy = metric.clone().cuda()
        self.test_accuracy = metric.clone().cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def step(self, x, y) -> Dict[str, torch.Tensor]:
        if self.self_supervised:
            z1, z2 = self.net.shared_step(x)
            logits = z1
            loss = self.loss(z1, z2)
        else:
            logits = self(x)
            if logits.shape[-1] > 1:
                loss = self.loss(self.final_nl(logits, dim=-1), y)
            else:
                logits = logits.ravel()
                loss = self.loss(self.final_nl(logits), y)
        return {"logits": logits, "loss": loss, "y": y, "x": x}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        out = self.step(x, y)
        # opt = self.optimizers()
        # opt.zero_grad()
        # self.manual_backward(out["loss"])
        # opt.step()
        return out

    def training_step_end(self, out):
        self.train_accuracy(self.final_nl(out["logits"], dim=-1), out["y"])
        self.log_dict(
            {
                "train_acc": self.train_accuracy,
                "train_loss": out["loss"].mean(),
            },
            on_step=True,
            on_epoch=False
        )
        return out["loss"].mean()

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        out = self.step(x, y)

        return out
    
    def validation_step_end(self, out):
        self.val_accuracy(self.final_nl(out["logits"], dim=-1), out["y"])
        self.log_dict(
            {
                "val_acc": self.val_accuracy,
                "val_loss": out["loss"].mean(),
            },
        )
        return {
            "image": out["x"],
            "y_true": out["y"],
            "logits": out["logits"],
            "val_loss": out["loss"].mean(),
        }

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        out = self.step(x, y)
        return out

    def test_step_end(self, out):
        self.test_accuracy(self.final_nl(out["logits"], dim=-1), out["y"])
        self.log_dict(
            {
                "test_acc": self.test_accuracy,
                "test_loss": out["loss"].mean(),
            },
        )
        return {
            "image": out["x"],
            "y_true": out["y"],
            "logits": out["logits"],
            "val_loss": out["loss"].mean(),
        }

    def on_validation_epoch_end(self):                                                      # def validation_epoch_end(self, outputs: List[Any]) -> None:
        if hasattr(self.net, "encoder"):
            layer = self.net.encoder.layer4
        else:
            layer = self.net.layer4
        integrated_gradients = GuidedGradCam(self, layer)
        batch_size = self.cfg.data.datamodule.batch_size.val
        images, images_feat_viz = [], []
        for output_element in iterate_elements_in_batches(
            self.validation_step_outputs, batch_size, self.cfg.logging.n_elements_to_log    # outputs, batch_size, self.cfg.logging.n_elements_to_log
        ):  
            rendered_image = render_images(
                output_element["image"],
                autoshow=False,
                normalize=self.cfg.logging.normalize_visualization)
            caption = f"y_pred: {output_element['logits'].argmax()}  [gt: {output_element['y_true']}]"  # noqa
            images.append(
                wandb.Image(
                    rendered_image,
                    caption=caption,
                )
            )

            # Add gradient visualization
            try:
                attributions_ig_nt = integrated_gradients.attribute(
                    output_element["image"].unsqueeze(0),
                    target=output_element["y_true"])
                vz = viz.visualize_image_attr(
                    np.transpose(attributions_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0)),
                    np.transpose(output_element["image"].cpu().detach().numpy(), (1, 2, 0)),
                    method='blended_heat_map',
                    show_colorbar=True,
                    use_pyplot=False,
                    # sign='positive',
                    outlier_perc=0)
                images_feat_viz.append(
                    wandb.Image(
                        vz[0],
                        caption=caption,
                    ))
                plt.close(vz[0])
            except:
                print("Failed to process attribution")
        self.logger.experiment.log({"Validation Images": images}, step=self.global_step)
        self.logger.experiment.log(
            {"Validation Images Viz": images_feat_viz},
            step=self.global_step)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        batch_size = self.cfg.data.datamodule.batch_size.test

        images = []
        images_feat_viz = []

        # integrated_gradients = IntegratedGradients(self.forward)
        # noise_tunnel = NoiseTunnel(integrated_gradients)
        import pdb;pdb.set_trace()
        self.logger.experiment.log({"Test Images": images}, step=self.global_step)
        return  # Don't need this stuff below vvvv

        for output_element in iterate_elements_in_batches(
            outputs, batch_size, self.cfg.logging.n_elements_to_log
        ):

            #import pdb; pdb.set_trace()
            attributions_ig_nt = noise_tunnel.attribute(output_element["image"].unsqueeze(0), nt_samples=50,
                                                        nt_type='smoothgrad_sq', target=output_element["y_true"],
                                                        internal_batch_size=50)
            vz = viz.visualize_image_attr(np.transpose(attributions_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(output_element["image"].cpu().detach().numpy(), (1, 2, 0)),
                                          method='blended_heat_map', show_colorbar=True, sign='positive', outlier_perc=1)

            rendered_image = render_images(output_element["image"], autoshow=False)
            caption = f"y_pred: {output_element['logits'].argmax()}  [gt: {output_element['y_true']}]"
            images.append(
                wandb.Image(
                    rendered_image,
                    caption=caption,
                )
            )
            images_feat_viz.append(
                wandb.Image(
                    vz[0],
                    caption=caption,
                ))
            plt.close(vz[0])
        self.logger.experiment.log({"Test Images": images}, step=self.global_step)
        self.logger.experiment.log({"Test Images Feature Viz": images_feat_viz}, step=self.global_step)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        if hasattr(self.cfg.optim, "exclude_bn_bias") and \
                self.cfg.optim.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.cfg.optim.optimizer.weight_decay)
            print("Warning: Excluding-biases-from-weight-decay is not properly implemented yet.")
            params = self.parameters()
        else:
            params = self.parameters()

        opt = hydra.utils.instantiate(
            self.cfg.optim.optimizer, params=params, weight_decay=self.cfg.optim.optimizer.weight_decay
        )
        
        if not self.cfg.optim.use_lr_scheduler:
            return opt

        # Handle schedulers if requested
        if 0:  # Need to fix this. self.cfg.optim.lr_scheduler.warmup_steps:
            # Right now this is specific to SimCLR
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    opt,
                    linear_warmup_decay(
                        self.cfg.optim.lr_scheduler.warmup_steps,
                        self.cfg.optim.lr_scheduler.total_steps,
                        cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            lr_scheduler = self.cfg.optim.lr_scheduler
        scheduler = hydra.utils.instantiate(lr_scheduler, optimizer=opt)
        return [opt], [scheduler]

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]
