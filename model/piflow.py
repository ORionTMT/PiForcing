from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from model.base import SelfForcingModel
import torch.nn as nn
from utils.wan_wrapper import WanDiffusionWrapper


class PiFlowID(SelfForcingModel):
    """
    Minimal Pi-Flow style speed-matching for Self-Forcing video.

    Key ideas for a minimal viable integration:
    - One student eval at time s to obtain a fixed policy proxy (x0_s).
    - Analytic sub-steps (no net calls) to integrate from s -> t using v_theta(x,t) ≈ (x - x0_s) / sigma(t).
    - One teacher (real_score) eval at (x_t, t) to obtain x0_t, then convert to flow v_t.

    Notes:
    - This is a pragmatic approximation to the original pi-Flow (no GMM policy). It respects the
      "one net eval per segment + many analytic sub-steps" spirit for quick prototyping.
    - For full pi-Flow, the student head should output policy parameters; this prototype uses x0_s as a proxy.
    """

    def __init__(self, args, device):
        super().__init__(args, device)

        # Rollout/block settings (kept consistent with DMD defaults)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            # teacher (real_score) is inference-only in this loss; no need to checkpoint

        # Pi-Flow segment hyperparameters
        self.pi_num_substeps = getattr(args, "pi_num_substeps", 64)  # analytic sub-steps within [s -> t]
        self.pi_use_last_step = getattr(args, "pi_use_last_step", True)  # use the current [s->t] in rollout

        # Training time / schedule controls
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.real_guidance_scale = getattr(args, "real_guidance_scale", getattr(args, "guidance_scale", 3.0))

        # Prepare alphas if provided by scheduler
        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

        # Lazy-initialized pipeline for generator rollout
        self.inference_pipeline = None

    def _initialize_models(self, args, device):
        """
        Override to avoid instantiating a full fake_score WAN model, which Pi-Flow does not use.
        Keep generator (causal), real_score (teacher, non-causal), text encoder, and VAE.
        Provide a tiny trainable placeholder for fake_score so the trainer/optimizer remain intact.
        """
        from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")

        # causal student
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        # teacher (non-causal WAN)
        self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
        self.real_score.model.requires_grad_(False)

        # tiny placeholder critic to keep optimizer/versioning paths simple
        class _TinyCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, *args, **kwargs):
                # Return shapes compatible with caller: (_, pred)
                return None, None

        self.fake_score = _TinyCritic()
        self.fake_score.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _sigma_from_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map (possibly fractional) timesteps t to sigmas using the scheduler's lookup.
        Shape: t [N] or [...]; returns sigma with the same leading shape.
        """
        original_dtype = t.dtype
        sigmas, timesteps = map(
            lambda x: x.double().to(t.device),
            [self.scheduler.sigmas, self.scheduler.timesteps]
        )
        t_flat = t.reshape(-1).double()
        # nearest neighbor over scheduler timesteps
        tid = torch.argmin((timesteps.unsqueeze(0) - t_flat.unsqueeze(1)).abs(), dim=1)
        sigma = sigmas[tid].to(original_dtype)
        return sigma.reshape(t.shape)

    @torch.no_grad()
    def _integrate_from_s_to_t(self, x_s: torch.Tensor, x0_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                               num_substeps: int) -> torch.Tensor:
        """
        Analytically integrate dx/dτ = v_theta(x, τ) with v_theta(x, τ) ≈ (x - x0_s) / sigma(τ),
        using fixed x0_s from student eval at time s. Euler steps in timestep domain.

        Inputs:
            x_s: [B, F, C, H, W] latent at time s
            x0_s: [B, F, C, H, W] student-predicted clean at time s (policy proxy)
            s: [B, F] or [B] timestep ids
            t: [B, F] or [B] timestep ids (t < s)
        Output:
            x_t: [B, F, C, H, W]
        """
        # broadcast s, t to [B, F]
        if t.ndim == 1:
            t = t[:, None].repeat(1, x_s.shape[1])
        if s.ndim == 1:
            s = s[:, None].repeat(1, x_s.shape[1])

        # scalar dt per sample (uniform partition in the scheduler timestep domain)
        dt = (s - t) / float(num_substeps)
        x = x_s.clone()
        cur = s.clone()
        for _ in range(num_substeps):
            sigma_tau = self._sigma_from_t(cur)
            # v_theta = (x - x0_s) / sigma_tau
            v_theta = (x - x0_s) / sigma_tau[:, :, None, None, None]
            x = x - dt[:, :, None, None, None] * v_theta
            cur = cur - dt
        return x

    def compute_piflow_loss(
        self,
        original_latent: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        denoised_timestep_from: Optional[int],
        denoised_timestep_to: Optional[int],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Perform one Pi-Flow (π-ID) segment: pick (s -> t),
        - one student eval at (x_s, s) to get x0_s,
        - integrate analytically to x_t,
        - one teacher eval at (x_t, t) to get x0_t and thus v_t,
        - match v_theta(x_t, t) to v_t.
        """
        device = original_latent.device
        bsz, num_frames = original_latent.shape[:2]

        # Choose (s -> t)
        if self.ts_schedule and denoised_timestep_from is not None and denoised_timestep_to is not None:
            # Use the rollout's denoised step boundary for tighter matching
            s_scalar = float(denoised_timestep_from)
            t_scalar = float(denoised_timestep_to)
        else:
            # Fallback: uniform segment in [min_step, max_step]
            s_scalar = float(torch.randint(self.min_step + 1, self.max_step, (1,), device=device).item())
            # make a small step towards 0
            t_scalar = max(self.min_step, s_scalar - max(1.0, (self.max_step - self.min_step) / 50.0))

        # Sample noise and obtain x_s = add_noise(x0, s)
        noise = torch.randn_like(original_latent)
        s_tensor = torch.full((bsz, num_frames), int(s_scalar), device=device, dtype=torch.long)
        x_s = self.scheduler.add_noise(
            original_latent.flatten(0, 1),
            noise.flatten(0, 1),
            s_tensor.flatten(0, 1)
        ).unflatten(0, (bsz, num_frames))

        # Student one-eval at (x_s, s) to get x0_s (policy proxy)
        with torch.set_grad_enabled(True):
            _, x0_s = self.generator(
                noisy_image_or_video=x_s,
                conditional_dict=conditional_dict,
                timestep=s_tensor
            )

        # Analytic integration from s -> t with fixed x0_s
        t_tensor = torch.full((bsz,), float(t_scalar), device=device, dtype=torch.float32)
        s_tensor_float = torch.full((bsz,), float(s_scalar), device=device, dtype=torch.float32)
        # Convert to same dtype as scheduler timesteps (int64-like indices) but we keep float for interpolation
        x_t = self._integrate_from_s_to_t(x_s, x0_s, s_tensor_float, t_tensor, self.pi_num_substeps)

        # Teacher one-eval at (x_t, t) to get x0_t, then flow v_t
        t_tensor_long = torch.full((bsz, num_frames), int(t_scalar), device=device, dtype=torch.long)
        with torch.no_grad():
            # CFG over real_score (teacher)
            _, x0_real_cond = self.real_score(
                noisy_image_or_video=x_t,
                conditional_dict=conditional_dict,
                timestep=t_tensor_long
            )
            _, x0_real_uncond = self.real_score(
                noisy_image_or_video=x_t,
                conditional_dict=unconditional_dict,
                timestep=t_tensor_long
            )
            x0_teacher = x0_real_cond + (x0_real_cond - x0_real_uncond) * self.real_guidance_scale

            v_t = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=x0_teacher.flatten(0, 1),
                xt=x_t.flatten(0, 1),
                timestep=t_tensor_long.flatten(0, 1)
            ).unflatten(0, (bsz, num_frames))

        # Student v_theta at (x_t, t) from the s-policy: (x_t - x0_s) / sigma(t)
        sigma_t = self._sigma_from_t(t_tensor)  # [B]
        v_theta = (x_t - x0_s) / sigma_t[:, None, None, None, None]

        # Speed matching loss
        loss = F.mse_loss(v_theta.double(), v_t.double(), reduction="mean")

        log_dict = {
            "piflow_timestep_s": torch.tensor(s_scalar, device=device),
            "piflow_timestep_t": torch.tensor(t_scalar, device=device),
            "piflow_v_norm": torch.mean(torch.abs(v_theta)).detach(),
        }
        return loss, log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Unroll the causal generator briefly to obtain a student x0 (trajectory endpoint),
        then perform one Pi-Flow segment speed matching.
        """
        # Use the existing self-forcing rollout to simulate inference-like latents
        pred_latent, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )

        loss, piflow_logs = self.compute_piflow_loss(
            original_latent=pred_latent,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )
        return loss, piflow_logs

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        No critic is used in minimal Pi-Flow prototype. Return zero loss.
        """
        dummy = torch.zeros((), device=self.device, dtype=self.dtype)
        return dummy, {"critic_timestep": torch.zeros_like(dummy)}

