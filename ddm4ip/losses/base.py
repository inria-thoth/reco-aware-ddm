import abc
import contextlib
import dataclasses
from enum import Enum
from typing import Any, Mapping
import torch

from ddm4ip.data.base import Batch
from ddm4ip.trainers.base import BaseTrainer


class AbstractLoss(abc.ABC):
    def __init__(
        self,
        has_val_loss: bool,
        n_accum_steps: int = 1,
        gs_enabled: bool = False,
    ):
        self.has_val_loss = has_val_loss
        self.n_accum_steps = n_accum_steps
        self.cur_n_accum_steps = 0
        self.grad_scaler = torch.amp.grad_scaler.GradScaler(enabled=gs_enabled)

    @abc.abstractmethod
    def __call__(self, trainer: 'BaseTrainer', batch: Batch) -> Mapping[str, float | torch.Tensor]:
        pass

    @abc.abstractmethod
    def val_loss(self, trainer: 'BaseTrainer', batch: Batch) -> Mapping[str, float | torch.Tensor]:
        pass

    def is_full_step_complete(self) -> bool:
        return self.cur_n_accum_steps == 0

    @abc.abstractmethod
    def load_state_dict(self, state_dict):
        pass

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        pass

    @contextlib.contextmanager
    def optimizer_step(
        self,
        optimizer,
        scheduler,
        module,
        grad_scaler: torch.amp.grad_scaler.GradScaler | None,
    ):
        """Context manager that runs a backward inside it and steps on exit
        (every `n_accum_steps` calls)."""
        do_optim_step = (self.cur_n_accum_steps + 1) % self.n_accum_steps == 0
        raised = False
        try:
            if isinstance(module, torch.nn.parallel.DistributedDataParallel) and not do_optim_step:
                with module.no_sync():
                    yield optimizer
            else:
                yield optimizer
        except BaseException:
            # Do not step the optimizer if the body failed: re-raise the
            # original error instead.
            raised = True
            raise
        finally:
            if do_optim_step and not raised:
                skip_lr_sched = False
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                    grad_scaler.step(optimizer)
                    prev_scale = grad_scaler.get_scale()
                    grad_scaler.update()
                    skip_lr_sched = prev_scale != grad_scaler.get_scale()
                else:
                    optimizer.step()
                if scheduler is not None and not skip_lr_sched:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            self.cur_n_accum_steps = (self.cur_n_accum_steps + 1) % self.n_accum_steps


class OptState(Enum):
    INNER = 1
    OUTER = 2


@dataclasses.dataclass
class OptimizationState:
    current_steps: int
    state: OptState
    loss_dict: dict[str, float] = dataclasses.field(default_factory=dict)

    def increment_state(self):
        if self.state == OptState.INNER:
            self.state = OptState.OUTER
        elif self.state == OptState.OUTER:
            self.state = OptState.INNER
        else:
            raise NotImplementedError(self.state)
        self.current_steps = 0