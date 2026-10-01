import deepinv
import torch


class DPIRSolver(torch.nn.Module):
    """DPIR: half-quadratic splitting with a denoiser prior (DRUNet by default).

    Only the last `grad_iters` of the `max_iter` iterations are run with
    gradients enabled. With `sat_mask`, the data term is weighted by a
    per-channel reliability mask: pixels outside (sat_lo, sat_hi) get weight
    `sat_weight` and target `sat_target`, and the weighted proximal step is
    solved by conjugate gradient.
    """

    SIGMA_MAX = 49.0 / 255.0
    LAMB = 1 / 0.23

    def __init__(
        self,
        noise_level: float,
        grad_iters: int | None,
        prior_model: torch.nn.Module | None = None,
        max_iter: int = 8,
        compile: bool = False,
        sat_mask: bool = False,
        sat_hi: float = 0.99,
        sat_lo: float = 0.0,
        sat_cg_iters: int = 24,
        sat_cg_tol: float = 1e-5,
        sat_weight: float = 0.05,
        sat_target: float = 1.0,
    ):
        super().__init__()
        self.noise_level = noise_level
        self.sat_mask = bool(sat_mask)
        self.sat_hi = float(sat_hi)
        self.sat_lo = float(sat_lo)
        self.sat_cg_iters = int(sat_cg_iters)
        self.sat_cg_tol = float(sat_cg_tol)
        self.sat_weight = float(sat_weight)
        self.sat_target = float(sat_target)
        self._cfg_max_iter = max_iter
        self.sigma_seq, self.stepsize_seq, self.max_iter = self.get_DPIR_params()
        if grad_iters is None or grad_iters > self.max_iter:
            grad_iters = self.max_iter
        self.grad_iters = list(range(self.max_iter - grad_iters, self.max_iter))
        if prior_model is None:
            self.prior_net = deepinv.models.DRUNet()
        else:
            self.prior_net = prior_model
        self.prior_net.eval()
        self.prior_net.requires_grad_(False)
        # Only the denoiser is compiled. The recompile limit is raised above the
        # number of distinct DRUNet shapes so that it never falls back to eager.
        if compile:
            import torch._dynamo as _dynamo
            for _attr in ("recompile_limit", "cache_size_limit"):
                if hasattr(_dynamo.config, _attr):
                    setattr(_dynamo.config, _attr, 64)
            self.prior_net.compile(dynamic=True)

    def get_DPIR_params(self):
        """Denoiser noise level and step size of each iteration."""
        max_iter = self._cfg_max_iter
        s1 = self.SIGMA_MAX
        # The terminal denoiser level is floored like the step size, so that a
        # noise-free problem (noise_level=0) keeps a finite schedule.
        s2 = max(0.01, self.noise_level)
        sigma_denoiser = torch.logspace(
            torch.log10(torch.tensor(s1, dtype=torch.float32)),
            torch.log10(torch.tensor(s2, dtype=torch.float32)),
            steps=max_iter,
            dtype=torch.float32,
            device="cuda",)
        stepsize = (sigma_denoiser / max(0.01, self.noise_level)) ** 2
        return sigma_denoiser, self.LAMB * stepsize, max_iter

    @staticmethod
    def _sr_factor(physics) -> int:
        """Super-resolution factor of the physics (1 = no downsampling)."""
        try:
            return int(getattr(physics, "factor", 1) or 1)
        except (TypeError, ValueError):
            return 1

    def forward(self, y, physics, **kwargs):
        factor = self._sr_factor(physics)
        if factor > 1:
            # Super-resolution: the iterate lives in the high-resolution domain
            # and is initialized by bicubic upsampling of the measurement.
            x = torch.nn.functional.interpolate(
                y, scale_factor=factor, mode="bicubic", align_corners=False
            )
        else:
            x = y.clone()
        sat_w = sat_t = None
        if self.sat_mask:
            M = ((y < self.sat_hi) & (y > self.sat_lo)).to(y.dtype)
            sat_w = M + self.sat_weight * (1.0 - M)
            sat_t = M * y + self.sat_target * (1.0 - M)
        for it in range(self.max_iter):
            with torch.set_grad_enabled(mode=it in self.grad_iters):
                x = self.run_one_iteration(
                    y=y,
                    physics=physics,
                    x=x,
                    step_size=self.stepsize_seq[it],
                    g_param=self.sigma_seq[it],
                    sat_w=sat_w,
                    sat_t=sat_t,
                )
        return x

    def _prox_weighted_cg_step(self, x, physics, gamma, W, T):
        """Weighted data-fidelity proximal step, solved with CG per sample:
            argmin_z (gamma/2)||W^{1/2}(A z - T)||^2 + (1/2)||z - x||^2,
        i.e. (A^T W A + (1/gamma) I) z = A^T W T + (1/gamma) x, warm-started at x."""
        inv_gamma = 1.0 / gamma
        eps = 1e-12

        def _dot(a, b):
            return (a * b).sum(dim=(-3, -2, -1), keepdim=True)

        def matvec(v):
            return physics.A_adjoint(W * physics.A(v)) + inv_gamma * v

        b = physics.A_adjoint(W * T) + inv_gamma * x
        z = x
        r = b - matvec(z)
        p = r.clone()
        rs_old = _dot(r, r)
        for _ in range(self.sat_cg_iters):
            Ap = matvec(p)
            alpha = rs_old / (_dot(p, Ap) + eps)
            z = z + alpha * p
            r = r - alpha * Ap
            rs_new = _dot(r, r)
            if bool((rs_new.sqrt().max() < self.sat_cg_tol)):
                break
            p = r + (rs_new / (rs_old + eps)) * p
            rs_old = rs_new
        return z

    def run_one_iteration(self, y, physics, x, **kwargs):
        sat_w = kwargs.get("sat_w", None)
        sat_t = kwargs.get("sat_t", None)
        if sat_w is not None:
            z = self._prox_weighted_cg_step(x, physics, kwargs["step_size"], sat_w, sat_t)
        else:
            # deepinv's implicit-backward least-squares solver drops the gradient
            # with respect to a non-leaf filter, so it is disabled.
            physics.implicit_backward_solver = False
            z = physics.prox_l2(x, y, kwargs["step_size"])
        x = self.prior_net(z, kwargs["g_param"])
        return x
