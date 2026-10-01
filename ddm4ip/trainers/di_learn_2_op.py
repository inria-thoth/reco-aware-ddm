import warnings

import torch

from ddm4ip.data.base import Batch, Datasplit
from ddm4ip.losses import DiffInstructOnXandY
from ddm4ip.networks import init_kernel_net
from ddm4ip.networks.blur_nets import DegradationNetwork
from ddm4ip.trainers.diffinstruct_learn_op import DiffinstructOpTrainer
from ddm4ip.trainers.plots import plot_explicit_kernels, plot_implicit_kernels, plot_spacevarying_kernels
from ddm4ip.utils import distributed
from ddm4ip.utils.easy_dict import print_module_summary


class DI2OpTrainer(DiffinstructOpTrainer):
    def init_models(self, cfg, dsets, device):
        dataset = self.get_any_dset(dsets)
        assert dataset is not None

        # Initialize the kernel NN
        kernel_nn = init_kernel_net(cfg["models"]["kernel"]).to(device)
        kernel_nn_fake_inputs = {
            "clean_img": torch.zeros([self.batch_size, *dataset.clean_img_size], device=device),
            "sigma": torch.zeros([self.batch_size], device=device),
            "conditioning": torch.zeros([self.batch_size, dataset.clean_conditioning_channels, *dataset.clean_img_size[1:]], device=device)
        }
        kernel_nn_fake_output: torch.Tensor = kernel_nn(**kernel_nn_fake_inputs)
        self.corrupt_img_size = (kernel_nn_fake_output.shape[1], kernel_nn_fake_output.shape[2], kernel_nn_fake_output.shape[3])

        x_img_size = dataset.clean_img_size
        y_img_size = self.corrupt_img_size

        # Initialize UNet for DiX. Takes clean images as input
        prtr_flow_nn_x, aux_flow_nn_x = self._init_pretrained_unet(
            cfg["models"]["pretrained_flow_x"],
            img_size=x_img_size,
            cond_ch=dataset.clean_conditioning_channels,
            label_dim=dataset.label_dim,
        )
        prtr_flow_nn_x.to(device)
        aux_flow_nn_x.to(device)

        # Initialize UNet for DiY. Takes corrupted images as input
        prtr_flow_nn_y, aux_flow_nn_y = self._init_pretrained_unet(
            cfg["models"]["pretrained_flow_y"],
            img_size=y_img_size,
            cond_ch=dataset.corrupt_conditioning_channels,
            label_dim=dataset.label_dim,
        )
        prtr_flow_nn_y.to(device)
        aux_flow_nn_y.to(device)

        if distributed.get_rank() == 0:
            print("-----------------------AUXILIARY FLOW NN X------------------------")
            cond_ch = dataset.clean_conditioning_channels
            print_module_summary(aux_flow_nn_x, {
                "x": torch.zeros([self.batch_size, *dataset.clean_img_size], device=device),
                "sigma": torch.ones([self.batch_size], device=device),
                "conditioning": torch.zeros([self.batch_size, cond_ch, *dataset.clean_img_size[1:]], device=device)
            }, max_nesting=2)
            print("-----------------------AUXILIARY FLOW NN Y------------------------")
            cond_ch = dataset.corrupt_conditioning_channels
            print_module_summary(aux_flow_nn_y, {
                "x": torch.zeros([self.batch_size, *self.corrupt_img_size], device=device),
                "sigma": torch.ones([self.batch_size], device=device),
                "conditioning": torch.zeros([self.batch_size, cond_ch, *self.corrupt_img_size[1:]], device=device)
            }, max_nesting=2)
            print("-------------------------KERNEL NN-------------------------")
            print_module_summary(kernel_nn, kernel_nn_fake_inputs, max_nesting=2)

        models = {
            "kernel_nn": kernel_nn,
            "prtr_flow_nn_x": prtr_flow_nn_x,
            "aux_flow_nn_x": aux_flow_nn_x,
            "prtr_flow_nn_y": prtr_flow_nn_y,
            "aux_flow_nn_y": aux_flow_nn_y,
        }
        return models

    def init_loss(self, cfg, models):
        match cfg["loss"].get("name"):
            case None:
                raise ValueError("Could not instantiate un-named loss.")
            case "di_x_y":
                return DiffInstructOnXandY(cfg, **models)  # pyright: ignore[reportArgumentType]
            case unsupported:
                raise ValueError(f"Loss name {unsupported} invalid. Expected 'di_x_y'.")

    def make_plots(self, cfg, dsets, dloaders, models, tb_writer, global_step):
        if self.plot_batch is None:
            # This ensures we always use the same batch for plotting.
            if (ds := dsets.get(Datasplit.TEST)) is None:
                warnings.warn("Using training batch for plotting.")
                ds = dsets[Datasplit.TRAIN]
                assert ds is not None
            # Don't use the dataloader to get around data-sampling artefacts.
            # messing around with indexes to get decent images for plotting
            self.plot_batch = Batch.collate_fn([ds[i] for i in range(self.batch_size)])

        kernel_nn = models["kernel_nn"]
        assert isinstance(kernel_nn, DegradationNetwork)
        dset = self.get_any_dset(dsets)
        assert self.corrupt_img_size is not None

        p2, p2_ker = None, None
        try:
            p2, p2_ker = plot_spacevarying_kernels(
                dset.corruption, dset.corrupt_img_size[-1], kernel_nn, self.batch_size, self.device
            )
        except TypeError:
            pass
        if p2 is None:
            p2, p2_ker = plot_explicit_kernels(dset.corruption, kernel_nn, self.plot_batch, self.batch_size, self.device)
        if p2 is not None and distributed.get_rank() == 0:
            self.log_image(p2, 'eval/explicit-kernels', tb_writer)
        if p2_ker is not None and distributed.get_rank() == 0:
            tb_writer.add_tensor('eval/kernel', p2_ker, self.global_step)
        p3 = plot_implicit_kernels(self.plot_batch, kernel_nn, loss=self.loss_optim)
        if p3 is not None and distributed.get_rank() == 0:
            self.log_image(p3, 'eval/implicit-kernels', tb_writer)
