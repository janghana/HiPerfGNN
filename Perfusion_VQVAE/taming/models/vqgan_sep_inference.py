import matplotlib.pyplot as plt
import wandb
import math

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from taming import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer3 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.data.torchinterp1d import interp1d

class VQModel_SEP_Inference(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="dsc_signal",
                 gt_key="dsc_signal",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,
                 ):
        super().__init__()
        self.input_key = input_key
        self.gt_key = gt_key
        self.encoder = Encoder(**ddconfig)
        ddconfig2 = ddconfig.copy()
        ddconfig2["num_down"] = [1,2,2,1,5]
        self.decoder = Decoder(**ddconfig2)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv1d(ddconfig["z_channels"], ddconfig["z_channels"], 1)
        self.post_quant_conv = torch.nn.Conv1d(ddconfig["z_channels"], ddconfig["z_channels"], 1)

        self.original_time_interval = 1.5
        self.original_time_n_step = 60
        self.original_time_range = self.original_time_interval * self.original_time_n_step

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.train_log_dict = {}
        self.valid_log_dict = {}

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, return_continuous=False):
        h = self.encoder(x)
        h = self.quant_conv(h)
        z_q, emb_loss, info = self.quantize(h)
        if return_continuous:
            return h, z_q, emb_loss, info
        else:
            return z_q, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        return batch[k]

    def transform_input(self, x):
        random_time_interval = np.random.choice([1, 1.25, 1.5])
        random_time_n_step = np.random.choice([30, 35, 40, 45, 50, 55, 60])
        random_time_range = random_time_interval * random_time_n_step
        target_size = int(random_time_n_step//5)

        if self.original_time_interval != random_time_interval or self.original_time_n_step != random_time_n_step:
            batch_size, C, _ = x.shape
            random_original_time_n_step = math.ceil(random_time_range / self.original_time_range * self.original_time_n_step)
            random_original_time_range = random_original_time_n_step * self.original_time_interval
            random_original_time_error = (random_original_time_range - random_time_range) / random_original_time_range
            x = x[..., :random_original_time_n_step].unsqueeze(-1)
            grid = torch.linspace(0, 1 - random_original_time_error - 1/random_time_n_step, random_time_n_step, device=x.device).view(1, 1, random_time_n_step).expand(batch_size, -1, -1)
            grid = torch.stack([-torch.ones_like(grid), grid], dim=-1)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
            x = x.view(batch_size, C, random_time_n_step)

        return x, target_size

    def training_step(self, batch, batch_idx):
        input = self.get_input(batch, self.input_key)
        input_t, target_size = self.transform_input(input)
        xrec, qloss = self(input_t)

        aeloss, log_dict_ae = self.loss(qloss, input, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")

        for key in log_dict_ae:
            if key not in self.train_log_dict:
                self.train_log_dict[key] = []
            self.train_log_dict[key].append(log_dict_ae[key])
        return aeloss

    def training_epoch_end(self, outputs):
        if len(self.train_log_dict) > 0:
            for key in self.train_log_dict:
                self.log(key, sum(self.train_log_dict[key])/len(self.train_log_dict[key]), prog_bar=True, logger=True, on_step=False, on_epoch=True)
            self.train_log_dict = {}

    def validation_step(self, batch, batch_idx):
        input = self.get_input(batch, self.input_key)
        input_t, target_size = self.transform_input(input)
        xrec, qloss = self(input_t)

        aeloss, log_dict_ae = self.loss(qloss, input, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        for key in log_dict_ae:
            if key not in self.valid_log_dict:
                self.valid_log_dict[key] = []
            self.valid_log_dict[key].append(log_dict_ae[key])
        return aeloss

    def test_step(self, batch, batch_idx):
        input = self.get_input(batch, self.input_key)
        input_t, target_size = self.transform_input(input)
        xrec, qloss = self(input_t)
        aeloss, log_dict_ae = self.loss(
            qloss, input, xrec, 0, self.global_step,
            last_layer=self.get_last_layer(), split="test"
        )
        self.log("test_loss", aeloss, prog_bar=True)
        return {"test_loss": aeloss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("test_loss_epoch", avg_loss, prog_bar=True)

    def validation_epoch_end(self, outputs):
        if len(self.valid_log_dict) > 0:
            for key in self.valid_log_dict:
                self.log(key, sum(self.valid_log_dict[key])/len(self.valid_log_dict[key]), prog_bar=True, logger=True, on_step=False, on_epoch=True)
            self.valid_log_dict = {}

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def get_last_layer(self):
        return self.decoder.conv_out.weight
