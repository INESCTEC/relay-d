"""
model_runner.py
---------------
Multi-algorithm policy inference — no robomimic dependency.

Supported model types (auto-detected from checkpoint structure):

  bc_mlp       Robomimic BC MLP  — deterministic MLP, obs -> action
  bc_rnn       Robomimic BC RNN  — stateful LSTM; call reset() between episodes
  bc_gmm       Robomimic BC GMM  — Gaussian mixture; returns mean of top component
  diffusion    Diffusion Policy  — scaffold; subclass to implement denoising loop
  act          ACT               — scaffold; subclass to implement action chunking
  torch_script Any model exported with torch.jit.save() — generic obs-concat -> action

Detection priority:
  1. TorchScript file (torch.jit.load succeeds)
  2. Explicit 'model_type' key inside the checkpoint dict
  3. Robomimic weight-key heuristics  (rnn > gmm > mlp)
  4. Diffusion Policy key patterns    (noise_pred_net)
  5. ACT key patterns                 (model_state_dict + transformer)
  6. ValueError with instructions

Override auto-detection by passing model_type= to ModelRunner.
"""

from __future__ import annotations

import abc
import json
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from relay_d.utils.coloring_logger import logger


# ============================================================================
# Abstract base
# ============================================================================

class BasePolicy(abc.ABC):
    """Interface every policy backend must implement."""

    DEFAULT_OBS_KEYS: List[str] = [
        "object",
        "robot0_eef_pos",
        "robot0_eef_quat",
    ]

    def __init__(
        self,
        obs_keys: Optional[List[str]] = None,
        obs_norm_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        self.obs_keys: List[str] = obs_keys or list(self.DEFAULT_OBS_KEYS)
        self._obs_norm_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = obs_norm_stats or {}

    def _build_obs_vector(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate obs_dict values in self.obs_keys order, normalizing keys
        for which the checkpoint provides obs_normalization_stats."""
        parts = []
        for key in self.obs_keys:
            if key not in obs_dict:
                raise KeyError(
                    f"[{type(self).__name__}] Missing obs key '{key}'. "
                    f"Expected keys: {self.obs_keys}"
                )
            val = obs_dict[key].astype(np.float32)
            if val.ndim > 1:
                raise ValueError(
                    f"[{type(self).__name__}] obs key '{key}' has shape "
                    f"{val.shape} — looks like an image/pointcloud passthrough "
                    f"obs (see obs_config.yaml's 'source:' entries). This "
                    f"backend (model_runner.py) flattens every obs key into a "
                    f"single 1-D vector for low-dim-only policy networks with "
                    f"no vision encoder (_BCMLP/_BCRNN/_BCGMM/diffusion) — it "
                    f"cannot consume this key. Camera/pointcloud obs are only "
                    f"supported via --backend robomimic, which delegates to "
                    f"robomimic's own vision-capable RolloutPolicy."
                )
            if key in self._obs_norm_stats:
                offset, scale = self._obs_norm_stats[key]
                val = (val - offset) / scale
            parts.append(val)
        vec = np.concatenate(parts)
        expected = getattr(self, "_obs_dim", None)
        if expected is not None and vec.shape[0] != expected:
            breakdown = {k: obs_dict[k].shape[0] for k in self.obs_keys}
            available = {k: v.shape[0] for k, v in obs_dict.items() if k not in self.obs_keys}
            raise ValueError(
                f"\n[{type(self).__name__}] Observation dimension mismatch!\n"
                f"  Model expects : {expected} dims\n"
                f"  Assembled     : {vec.shape[0]} dims from keys: {breakdown}\n"
                f"  Gap           : {expected - vec.shape[0]} dims\n"
                f"  Available keys not in use: {available}\n"
                f"Hint: pass --obs-keys with the correct set, or check that obs_config.yaml "
                f"contains the keys used during training."
            )
        return vec

    @abc.abstractmethod
    def get_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Run one forward pass and return the action array."""

    def reset(self) -> None:
        """Reset stateful components (e.g. RNN hidden state). Call between episodes."""


# ============================================================================
# Network modules
# ============================================================================

class _BCMLP(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, layer_dims: List[int]):
        super().__init__()
        dims = [obs_dim] + list(layer_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.action_head = nn.Linear(layer_dims[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.mlp(obs))


class _BCRNN(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out, state = self.lstm(obs, state)          # obs: (1, 1, obs_dim)
        action = self.action_head(out.squeeze(1))   # -> (1, action_dim)
        return action, state


class _BCGMM(nn.Module):
    """MLP backbone + Gaussian Mixture heads. Returns mean of top-logit component."""

    def __init__(self, obs_dim: int, layer_dims: List[int], action_dim: int, num_modes: int):
        super().__init__()
        dims = [obs_dim] + list(layer_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.mean_head   = nn.Linear(layer_dims[-1], num_modes * action_dim)
        self.logits_head = nn.Linear(layer_dims[-1], num_modes)
        self.num_modes  = num_modes
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat   = self.mlp(obs)
        means  = self.mean_head(feat).view(-1, self.num_modes, self.action_dim)
        logits = self.logits_head(feat)
        best   = logits.argmax(dim=-1)
        return means[torch.arange(means.size(0)), best]   # (batch, action_dim)


# ============================================================================
# Diffusion Policy network modules (Chi et al. 2023 / robomimic)
# ============================================================================

class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=x.device).float() / max(half - 1, 1)
        )
        args = x.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class _Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Conv1dBlock(nn.Module):
    def __init__(self, inp: int, out: int, k: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, k, padding=k // 2),
            nn.GroupNorm(n_groups, out),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, k: int = 3, n_groups: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            _Conv1dBlock(in_ch,  out_ch, k, n_groups),
            _Conv1dBlock(out_ch, out_ch, k, n_groups),
        ])
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
        )
        self.residual_conv = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)   # (B, 2*C, 1)
        scale, shift = embed.chunk(2, dim=1)
        out = out * scale + shift
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class _ConditionalUnet1D(nn.Module):
    """UNet1d with global FiLM conditioning. Architecture mirrors Chi et al. (2023)."""

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int,
        down_dims: List[int],
        kernel_size: int,
        n_groups: int,
        down_has_downsample: List[bool],
        up_has_upsample: List[bool],
    ):
        super().__init__()
        dsed = diffusion_step_embed_dim
        cond_dim = global_cond_dim + dsed
        all_dims = [input_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.diffusion_step_encoder = nn.Sequential(
            _SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        self.down_modules = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(in_out):
            has_ds = down_has_downsample[i] if i < len(down_has_downsample) else False
            self.down_modules.append(nn.ModuleList([
                _ConditionalResidualBlock1D(dim_in,  dim_out, cond_dim, kernel_size, n_groups),
                _ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                _Downsample1d(dim_out) if has_ds else nn.Identity(),
            ]))

        mid_dim = down_dims[-1]
        self.mid_modules = nn.ModuleList([
            _ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            _ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])

        # Up path: one level per actual downsample; skip from corresponding down level
        self.up_modules = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            has_us = up_has_upsample[i] if i < len(up_has_upsample) else False
            self.up_modules.append(nn.ModuleList([
                _ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                _ConditionalResidualBlock1D(dim_in,      dim_in, cond_dim, kernel_size, n_groups),
                _Upsample1d(dim_in) if has_us else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            _Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0).expand(sample.shape[0])
        t_emb = self.diffusion_step_encoder(timestep)
        global_feature = torch.cat([t_emb, global_cond], dim=-1)

        h: List[torch.Tensor] = []
        x = sample
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        return self.final_conv(x)


class _DDPMScheduler:
    """Inline DDPM scheduler with squaredcos_cap_v2 beta schedule."""

    def __init__(
        self,
        num_train_timesteps: int = 100,
        num_inference_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
    ):
        self.clip_sample = clip_sample
        betas = self._make_betas(num_train_timesteps, beta_schedule)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Matches diffusers' DDPMScheduler.set_timesteps: anchor at 0, step up
        # by step_ratio, then reverse — NOT the same sequence as counting down
        # from num_train_timesteps-1 whenever step_ratio != 1 (e.g. T=100,
        # num_inference=10 gives [90,...,10,0] here vs [99,...,9] the other way).
        step_ratio = num_train_timesteps // num_inference_timesteps
        self.timesteps = torch.arange(0, num_train_timesteps, step_ratio).flip(0).long()

    @staticmethod
    def _make_betas(T: int, schedule: str, s: float = 0.008) -> torch.Tensor:
        if schedule == "squaredcos_cap_v2":
            steps = T + 1
            t = torch.linspace(0, T, steps, dtype=torch.float64) / T
            alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            return betas.clamp(0, 0.999).float()
        raise ValueError(f"Unknown beta_schedule '{schedule}'. Use 'squaredcos_cap_v2'.")

    def to(self, device: torch.device) -> "_DDPMScheduler":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.timesteps = self.timesteps.to(device)
        return self

    def step(
        self,
        noise_pred: torch.Tensor,
        t_cur: int,
        x_t: torch.Tensor,
        t_prev: int = -1,
    ) -> torch.Tensor:
        """Compute x_{t_prev} from x_{t_cur} and the predicted noise."""
        dev = x_t.device
        acp_t    = self.alphas_cumprod[t_cur]
        acp_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=dev)
        beta_t   = self.betas[t_cur]
        alpha_t  = self.alphas[t_cur]

        # Predict clean sample x_0
        pred_x0 = (x_t - (1 - acp_t).sqrt() * noise_pred) / acp_t.sqrt()
        if self.clip_sample:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

        # Posterior mean
        coef1 = acp_prev.sqrt() * beta_t / (1 - acp_t)
        coef2 = alpha_t.sqrt() * (1 - acp_prev) / (1 - acp_t)
        mean  = coef1 * pred_x0 + coef2 * x_t

        if t_prev >= 0:
            posterior_var = (beta_t * (1 - acp_prev) / (1 - acp_t)).clamp(min=1e-20)
            x_prev = mean + posterior_var.sqrt() * torch.randn_like(x_t)
        else:
            x_prev = mean

        return x_prev


# ============================================================================
# Concrete policies
# ============================================================================

class RobomimicBCMLPPolicy(BasePolicy):
    """
    Robomimic BC with MLP backbone. Layer count/widths are discovered from the
    checkpoint (via _discover_mlp_layer_dims) — actor_layer_dims may be any
    depth and need not be uniform across layers.

    Expected checkpoint keys:
      ckpt["model"]["nets"]["policy.nets.mlp._model.0.weight"]          (hidden_0, obs)
      ckpt["model"]["nets"]["policy.nets.mlp._model.2.weight"]          (hidden_1, hidden_0)
      ckpt["model"]["nets"]["policy.nets.mlp._model.{2k}.weight"]       (..., as many layers as present)
      ckpt["model"]["nets"]["policy.nets.decoder.nets.action.weight"]   (action, hidden_last)
      ckpt["action_normalization_stats"]["actions"]["scale" / "offset"]
    """

    def __init__(
        self,
        ckpt: dict,
        device: torch.device,
        obs_keys: Optional[List[str]] = None,
    ):
        super().__init__(obs_keys, obs_norm_stats=_load_obs_normalization(ckpt))
        nets = ckpt["model"]["nets"]
        self._device = device

        obs_dim    = nets["policy.nets.mlp._model.0.weight"].shape[1]
        layer_dims = _discover_mlp_layer_dims(nets, "policy.nets.mlp._model")
        action_dim = nets["policy.nets.decoder.nets.action.weight"].shape[0]

        self._obs_dim = obs_dim

        sm = ckpt.get("shape_metadata", {})
        ckpt_obs_keys = sm.get("all_obs_keys")
        if ckpt_obs_keys:
            self.obs_keys = list(ckpt_obs_keys)
            logger.info(f"[BC_MLP] obs_keys loaded from checkpoint: {self.obs_keys}")

        self._net = _BCMLP(obs_dim, action_dim, layer_dims).to(device)
        self._load_weights(nets)
        self._net.eval()
        self._scale, self._offset = _load_normalization(ckpt, action_dim, device)

        logger.info(f"[BC_MLP] obs={obs_dim} layer_dims={layer_dims} action={action_dim}")

    def _load_weights(self, nets: dict) -> None:
        state_dict = {}
        i = 0
        while f"policy.nets.mlp._model.{i}.weight" in nets:
            state_dict[f"mlp.{i}.weight"] = nets[f"policy.nets.mlp._model.{i}.weight"]
            state_dict[f"mlp.{i}.bias"]   = nets[f"policy.nets.mlp._model.{i}.bias"]
            i += 2
        state_dict["action_head.weight"] = nets["policy.nets.decoder.nets.action.weight"]
        state_dict["action_head.bias"]   = nets["policy.nets.decoder.nets.action.bias"]
        self._net.load_state_dict(state_dict)

    def get_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        obs = torch.from_numpy(self._build_obs_vector(obs_dict)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            action = self._net(obs) * self._scale + self._offset
        return action.squeeze(0).cpu().numpy()


class RobomimicBCRNNPolicy(BasePolicy):
    """
    Robomimic BC_RNN with LSTM backbone (stateful).

    Expected checkpoint keys:
      ckpt["model"]["nets"]["policy.nets.rnn._model.weight_ih_l0"]       (4*H, obs)
      ckpt["model"]["nets"]["policy.nets.rnn._model.weight_hh_l0"]       (4*H, H)
      ckpt["model"]["nets"]["policy.nets.rnn._model.bias_ih_l0"]
      ckpt["model"]["nets"]["policy.nets.rnn._model.bias_hh_l0"]
      ckpt["model"]["nets"]["policy.nets.decoder.nets.action.weight"]    (action, H)
      ckpt["action_normalization_stats"]["actions"]["scale" / "offset"]
    Multi-layer LSTMs: keys with _l1, _l2, ... are handled automatically.

    NOTE: If your checkpoint includes a pre-LSTM obs encoder (obs_nets.*), the
    LSTM input dimension will not equal obs_dim. You will need to subclass and
    override _load_weights() to incorporate the encoder layers.

    Call reset() at the start of each new episode.
    """

    def __init__(
        self,
        ckpt: dict,
        device: torch.device,
        obs_keys: Optional[List[str]] = None,
    ):
        super().__init__(obs_keys, obs_norm_stats=_load_obs_normalization(ckpt))
        nets = ckpt["model"]["nets"]
        self._device = device

        ih_l0      = nets["policy.nets.rnn._model.weight_ih_l0"]  # (4*H, obs_or_enc)
        hh_l0      = nets["policy.nets.rnn._model.weight_hh_l0"]  # (4*H, H)
        obs_dim    = ih_l0.shape[1]
        hidden_dim = hh_l0.shape[1]
        action_dim = nets["policy.nets.decoder.nets.action.weight"].shape[0]
        num_layers = sum(
            1 for k in nets
            if k.startswith("policy.nets.rnn._model.weight_ih_l")
        )

        self._obs_dim = obs_dim

        sm = ckpt.get("shape_metadata", {})
        ckpt_obs_keys = sm.get("all_obs_keys")
        if ckpt_obs_keys:
            self.obs_keys = list(ckpt_obs_keys)
            logger.info(f"[BC_RNN] obs_keys loaded from checkpoint: {self.obs_keys}")

        self._net = _BCRNN(obs_dim, hidden_dim, action_dim, num_layers).to(device)
        self._load_weights(nets, num_layers)
        self._net.eval()
        self._state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._scale, self._offset = _load_normalization(ckpt, action_dim, device)

        logger.info(
            f"[BC_RNN] obs={obs_dim} lstm_hidden={hidden_dim} "
            f"layers={num_layers} action={action_dim}"
        )

    def _load_weights(self, nets: dict, num_layers: int) -> None:
        sd = {}
        for l in range(num_layers):
            p = "policy.nets.rnn._model"
            sd[f"lstm.weight_ih_l{l}"] = nets[f"{p}.weight_ih_l{l}"]
            sd[f"lstm.weight_hh_l{l}"] = nets[f"{p}.weight_hh_l{l}"]
            sd[f"lstm.bias_ih_l{l}"]   = nets[f"{p}.bias_ih_l{l}"]
            sd[f"lstm.bias_hh_l{l}"]   = nets[f"{p}.bias_hh_l{l}"]
        sd["action_head.weight"] = nets["policy.nets.decoder.nets.action.weight"]
        sd["action_head.bias"]   = nets["policy.nets.decoder.nets.action.bias"]
        self._net.load_state_dict(sd)

    def get_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        obs = torch.from_numpy(self._build_obs_vector(obs_dict))
        obs = obs.unsqueeze(0).unsqueeze(0).to(self._device)   # (1, 1, obs_dim)
        with torch.no_grad():
            action, self._state = self._net(obs, self._state)
            action = action * self._scale + self._offset
        return action.squeeze(0).cpu().numpy()

    def reset(self) -> None:
        self._state = None
        logger.debug("[BC_RNN] LSTM hidden state reset.")


class RobomimicBCGMMPolicy(BasePolicy):
    """
    Robomimic BC_GMM (MLP + Gaussian Mixture output). Layer count/widths are
    discovered from the checkpoint (via _discover_mlp_layer_dims) —
    actor_layer_dims may be any depth and need not be uniform across layers;
    the GMM heads attach to the last discovered layer's width.

    Expected checkpoint keys:
      ckpt["model"]["nets"]["policy.nets.mlp._model.{2k}.weight"]          (as many layers as present)
      ckpt["model"]["nets"]["policy.nets.decoder.nets.mean.weight"]     (K*action, hidden_last)
      ckpt["model"]["nets"]["policy.nets.decoder.nets.mean.bias"]
      ckpt["model"]["nets"]["policy.nets.decoder.nets.logits.weight"]   (K, hidden_last)
      ckpt["model"]["nets"]["policy.nets.decoder.nets.logits.bias"]
      ckpt["action_normalization_stats"]["actions"]["scale" / "offset"]

    At inference: returns the mean of the highest-weight mixture component.
    Pass sample=True to ModelRunner to sample from the full distribution instead.

    NOTE: Key names follow robomimic's default BC_GMM config. If your
    checkpoint uses different names, subclass and override _load_weights().
    """

    def __init__(
        self,
        ckpt: dict,
        device: torch.device,
        obs_keys: Optional[List[str]] = None,
        sample: bool = False,
    ):
        super().__init__(obs_keys, obs_norm_stats=_load_obs_normalization(ckpt))
        nets = ckpt["model"]["nets"]
        self._device = device
        self._sample = sample

        obs_dim    = nets["policy.nets.mlp._model.0.weight"].shape[1]
        layer_dims = _discover_mlp_layer_dims(nets, "policy.nets.mlp._model")
        logits_w   = nets["policy.nets.decoder.nets.logits.weight"]   # (K, last_hidden)
        mean_w     = nets["policy.nets.decoder.nets.mean.weight"]     # (K*action, last_hidden)
        num_modes  = logits_w.shape[0]
        action_dim = mean_w.shape[0] // num_modes

        self._obs_dim = obs_dim

        sm = ckpt.get("shape_metadata", {})
        ckpt_obs_keys = sm.get("all_obs_keys")
        if ckpt_obs_keys:
            self.obs_keys = list(ckpt_obs_keys)
            logger.info(f"[BC_GMM] obs_keys loaded from checkpoint: {self.obs_keys}")

        self._net = _BCGMM(obs_dim, layer_dims, action_dim, num_modes).to(device)
        self._load_weights(nets)
        self._net.eval()
        self._scale, self._offset = _load_normalization(ckpt, action_dim, device)

        logger.info(
            f"[BC_GMM] obs={obs_dim} layer_dims={layer_dims} "
            f"modes={num_modes} action={action_dim} sample={sample}"
        )

    def _load_weights(self, nets: dict) -> None:
        state_dict = {}
        i = 0
        while f"policy.nets.mlp._model.{i}.weight" in nets:
            state_dict[f"mlp.{i}.weight"] = nets[f"policy.nets.mlp._model.{i}.weight"]
            state_dict[f"mlp.{i}.bias"]   = nets[f"policy.nets.mlp._model.{i}.bias"]
            i += 2
        state_dict["mean_head.weight"]   = nets["policy.nets.decoder.nets.mean.weight"]
        state_dict["mean_head.bias"]     = nets["policy.nets.decoder.nets.mean.bias"]
        state_dict["logits_head.weight"] = nets["policy.nets.decoder.nets.logits.weight"]
        state_dict["logits_head.bias"]   = nets["policy.nets.decoder.nets.logits.bias"]
        self._net.load_state_dict(state_dict)

    def get_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        obs = torch.from_numpy(self._build_obs_vector(obs_dict)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            action = self._net(obs) * self._scale + self._offset
        return action.squeeze(0).cpu().numpy()


class TorchScriptPolicy(BasePolicy):
    """
    Generic policy loaded from a TorchScript file (torch.jit.save()).

    The exported model must accept a 1-D float32 tensor (concatenated obs in
    obs_keys order) and return a 1-D float32 tensor (action). Example:

        model = torch.jit.script(MyPolicy())
        torch.jit.save(model, "policy.pt")

    obs_keys MUST be provided explicitly via --obs-keys; they cannot be
    inferred from a .pt file.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        obs_keys: Optional[List[str]] = None,
    ):
        super().__init__(obs_keys)
        self._device = device
        self._model = torch.jit.load(checkpoint_path, map_location=device)
        self._model.eval()
        logger.info(f"[TorchScript] Loaded: {checkpoint_path}")
        logger.info(f"[TorchScript] obs_keys: {self.obs_keys}")

    def get_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        obs = torch.from_numpy(self._build_obs_vector(obs_dict)).to(self._device)
        with torch.no_grad():
            action = self._model(obs)
        return action.cpu().numpy()


class DiffusionPolicy(BasePolicy):
    """
    Diffusion Policy (Chi et al. 2023) — scaffold.

    To enable, subclass and implement _build():

        class MyDiffusionPolicy(DiffusionPolicy):
            def _build(self, ckpt, device):
                self._noise_net = UNet1d(...)
                self._noise_net.load_state_dict(ckpt["model"]["noise_pred_net"])
                self._scheduler = DDPMScheduler(num_train_timesteps=100)

            def get_action(self, obs_dict):
                obs = ...
                noisy = torch.randn(1, pred_horizon, action_dim)
                for t in self._scheduler.timesteps:
                    noise_pred = self._noise_net(noisy, t, obs_cond)
                    noisy = self._scheduler.step(noise_pred, t, noisy).prev_sample
                return noisy[0, 0].cpu().numpy()  # first action of chunk

    Typical checkpoint keys:
      ckpt["model"]["noise_pred_net.*"]   UNet or Transformer noise predictor
      ckpt["model"]["obs_encoder.*"]      observation encoder
      ckpt["normalizer.*"]               input/output normalization

    Alternatively: export your trained policy with torch.jit.save() and use
    model_type='torch_script' to skip implementing this class.
    """

    def __init__(
        self,
        ckpt: dict,
        device: torch.device,
        obs_keys: Optional[List[str]] = None,
        action_horizon: Optional[int] = None,
    ):
        super().__init__(obs_keys, obs_norm_stats=_load_obs_normalization(ckpt))
        self._device = device
        # Deployment-time override for the receding-horizon control window;
        # None means "use the value baked into the checkpoint's training config".
        self._action_horizon_override = action_horizon
        self._build(ckpt, device)

    def _build(self, ckpt: dict, device: torch.device) -> None:
        raise NotImplementedError(
            "DiffusionPolicy._build() is not implemented.\n"
            "Subclass DiffusionPolicy, override _build() and get_action().\n"
            "See the class docstring for a step-by-step guide.\n"
            "Or export your model with torch.jit.save() and use model_type='torch_script'."
        )

    def get_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError("Implement get_action() in your DiffusionPolicy subclass.")


class ACTPolicy(BasePolicy):
    """
    Action Chunking with Transformers (Zhao et al. 2023) — scaffold.

    To enable, subclass and implement _build():

        class MyACTPolicy(ACTPolicy):
            def _build(self, ckpt, device):
                self._transformer = build_act_transformer(...)
                self._transformer.load_state_dict(ckpt["model_state_dict"])
                self._chunk_size = 100
                self._chunk: Optional[np.ndarray] = None
                self._step = 0

            def get_action(self, obs_dict):
                if self._chunk is None or self._step >= self._chunk_size:
                    obs = ...
                    self._chunk = self._transformer(obs).cpu().numpy()  # (chunk, action)
                    self._step = 0
                action = self._chunk[self._step]
                self._step += 1
                return action

            def reset(self):
                self._chunk = None
                self._step = 0

    Typical checkpoint keys:
      ckpt["model_state_dict"]["model.transformer.*"]
      ckpt["model_state_dict"]["model.encoder.*"]
      ckpt["model_state_dict"]["model.action_head.*"]

    Alternatively: export your trained policy with torch.jit.save() and use
    model_type='torch_script' to skip implementing this class.
    """

    def __init__(
        self,
        ckpt: dict,
        device: torch.device,
        obs_keys: Optional[List[str]] = None,
    ):
        super().__init__(obs_keys, obs_norm_stats=_load_obs_normalization(ckpt))
        self._device = device
        self._build(ckpt, device)

    def _build(self, ckpt: dict, device: torch.device) -> None:
        raise NotImplementedError(
            "ACTPolicy._build() is not implemented.\n"
            "Subclass ACTPolicy, override _build() and get_action().\n"
            "See the class docstring for a step-by-step guide.\n"
            "Or export your model with torch.jit.save() and use model_type='torch_script'."
        )

    def get_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError("Implement get_action() in your ACTPolicy subclass.")


class _RobomimicDiffusionPolicy(DiffusionPolicy):
    """
    Concrete DiffusionPolicy for checkpoints produced by robomimic's DiffusionPolicy algo.

    Architecture and hyperparameters are inferred from the checkpoint at load time.
    Uses EMA weights when available (preferred for inference).
    Implements DDPM with squaredcos_cap_v2 schedule.
    """

    def _build(self, ckpt: dict, device: torch.device) -> None:  # noqa: C901
        nets   = ckpt["model"]["nets"]
        ema    = ckpt["model"].get("ema")   # prefer EMA weights for inference
        prefix = "policy.noise_pred_net."

        def w(k: str) -> torch.Tensor:
            return nets[prefix + k]

        # ── infer architecture from weight shapes ──────────────────────────
        action_dim   = w("final_conv.1.weight").shape[0]
        dsed         = w("diffusion_step_encoder.1.weight").shape[1]
        cond_dim     = w("down_modules.0.0.cond_encoder.1.weight").shape[1]
        obs_cond_dim = cond_dim - dsed
        kernel_size  = w("down_modules.0.0.blocks.0.block.0.weight").shape[2]

        down_dims: List[int] = []
        for i in range(20):
            key = f"down_modules.{i}.0.blocks.0.block.0.weight"
            if (prefix + key) not in nets:
                break
            down_dims.append(nets[prefix + key].shape[0])

        n_down = len(down_dims)
        n_up   = n_down - 1
        down_has_ds = [(prefix + f"down_modules.{i}.2.conv.weight") in nets for i in range(n_down)]
        up_has_us   = [(prefix + f"up_modules.{i}.2.conv.weight")   in nets for i in range(n_up)]

        logger.info(
            f"[DiffusionPolicy] action_dim={action_dim}  obs_cond_dim={obs_cond_dim}  "
            f"down_dims={down_dims}  kernel={kernel_size}"
        )

        # ── read obs_keys / obs_dim from shape_metadata ────────────────────
        sm = ckpt.get("shape_metadata", {})
        ckpt_obs_keys   = sm.get("all_obs_keys")
        ckpt_obs_shapes = sm.get("all_shapes", {})

        if ckpt_obs_keys:
            ckpt_obs_dim = sum(ckpt_obs_shapes[k][0] for k in ckpt_obs_keys if k in ckpt_obs_shapes)
            if ckpt_obs_dim > 0 and obs_cond_dim % ckpt_obs_dim == 0:
                obs_horizon = obs_cond_dim // ckpt_obs_dim
            else:
                obs_horizon = 1
                logger.warning(
                    f"[DiffusionPolicy] obs_cond_dim={obs_cond_dim} not divisible by "
                    f"obs_dim={ckpt_obs_dim}; using obs_horizon=1"
                )
            if set(self.obs_keys) == set(BasePolicy.DEFAULT_OBS_KEYS):
                self.obs_keys = list(ckpt_obs_keys)
            logger.info(f"[DiffusionPolicy] obs_keys={self.obs_keys}  obs_horizon={obs_horizon}")
        else:
            obs_dim     = sum(1 for _ in self.obs_keys)   # fallback
            obs_horizon = max(1, obs_cond_dim // max(obs_dim, 1))

        # ── read diffusion hyperparameters from embedded config JSON ────────
        pred_horizon   = 16
        action_horizon = 8
        num_train_ts   = 100
        num_infer_ts   = 100
        beta_schedule  = "squaredcos_cap_v2"
        clip_sample    = True

        try:
            cfg = json.loads(ckpt.get("config", "{}"))
            algo = cfg.get("algo", {})

            hz = algo.get("horizon", {})
            if isinstance(hz, dict):
                obs_horizon    = hz.get("observation_horizon", obs_horizon)
                pred_horizon   = hz.get("prediction_horizon",  pred_horizon)
                action_horizon = hz.get("action_horizon",      action_horizon)

            ddpm = algo.get("ddpm", {})
            num_train_ts  = ddpm.get("num_train_timesteps",    num_train_ts)
            num_infer_ts  = ddpm.get("num_inference_timesteps", num_infer_ts)
            beta_schedule = ddpm.get("beta_schedule",          beta_schedule)
            clip_sample   = ddpm.get("clip_sample",            clip_sample)
        except Exception as exc:
            logger.warning(f"[DiffusionPolicy] Could not parse config JSON: {exc}")

        if self._action_horizon_override is not None:
            max_action_horizon = pred_horizon - (obs_horizon - 1)
            if not (1 <= self._action_horizon_override <= max_action_horizon):
                raise ValueError(
                    f"action_horizon override {self._action_horizon_override} is out of range "
                    f"[1, {max_action_horizon}] for pred_horizon={pred_horizon}, "
                    f"obs_horizon={obs_horizon} — the receding-horizon slice "
                    f"[obs_horizon-1 : obs_horizon-1+action_horizon] must stay within "
                    f"the predicted trajectory."
                )
            logger.info(
                f"[DiffusionPolicy] action_horizon overridden: "
                f"{action_horizon} (checkpoint) -> {self._action_horizon_override} (flag)"
            )
            action_horizon = self._action_horizon_override

        logger.info(
            f"[DiffusionPolicy] obs_horizon={obs_horizon}  pred_horizon={pred_horizon}  "
            f"action_horizon={action_horizon}  "
            f"num_train_ts={num_train_ts}  num_infer_ts={num_infer_ts}"
        )

        self._action_dim    = action_dim
        self._obs_horizon   = obs_horizon
        self._pred_horizon  = pred_horizon
        self._action_horizon = action_horizon

        # ── build network ───────────────────────────────────────────────────
        self._noise_pred_net = _ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_cond_dim,
            diffusion_step_embed_dim=dsed,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=8,
            down_has_downsample=down_has_ds,
            up_has_upsample=up_has_us,
        ).to(device)

        weight_src = ema if ema is not None else nets
        net_sd = {k[len(prefix):]: v for k, v in weight_src.items() if k.startswith(prefix)}
        self._noise_pred_net.load_state_dict(net_sd)
        self._noise_pred_net.eval()
        logger.info(f"[DiffusionPolicy] Weights loaded ({'EMA' if ema else 'non-EMA'}).")

        # ── build scheduler ─────────────────────────────────────────────────
        self._scheduler = _DDPMScheduler(
            num_train_timesteps=num_train_ts,
            num_inference_timesteps=num_infer_ts,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        ).to(device)

        # ── observation ring buffer & receding-horizon action queue ─────────
        self._obs_buf: deque = deque(maxlen=self._obs_horizon)
        self._action_queue: deque = deque()
        self.last_obs_cond = None  # debug introspection only, set on each replan

        # ── action un-normalization (identity if the checkpoint has none) ────
        self._scale, self._offset = _load_normalization(ckpt, action_dim, device)

    def get_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # Debug introspection only: cleared every call, set only on ticks
        # that actually replan below — lets a caller tell whether this
        # specific tick recomputed obs_cond or just popped a cached action.
        self.last_obs_cond = None

        # Track obs history every tick (even on ticks that replay a cached
        # action below) so the window is up to date whenever we next replan —
        # mirrors how robomimic's own rollout loop keeps feeding fresh obs in
        # every step regardless of the action queue's state.
        obs_vec = torch.from_numpy(
            self._build_obs_vector(obs_dict).astype(np.float32)
        ).to(self._device)

        if len(self._obs_buf) == 0:
            for _ in range(self._obs_horizon):
                self._obs_buf.append(obs_vec)
        else:
            self._obs_buf.append(obs_vec)

        if len(self._action_queue) == 0:
            # No cached actions left — replan: run the full DDPM denoise and
            # refill the queue with an action_horizon-length chunk, exactly
            # as robomimic's DiffusionPolicyUNet.get_action does. Between
            # replans, obs updates above are tracked but not acted on — the
            # trained policy assumes action_horizon ticks pass before the
            # next replan, and re-denoising every tick (the previous
            # behavior here) both wastes compute and introduces fresh DDPM
            # sampling noise every call instead of the validated
            # once-per-chunk noise draw.
            obs_cond = torch.stack(list(self._obs_buf), dim=0).flatten().unsqueeze(0)  # (1, H*obs_dim)
            self.last_obs_cond = obs_cond.detach().cpu().numpy()  # debug introspection only

            x = torch.randn(1, self._action_dim, self._pred_horizon, device=self._device)

            ts = self._scheduler.timesteps
            for i in range(len(ts)):
                t_cur  = int(ts[i].item())
                t_prev = int(ts[i + 1].item()) if i + 1 < len(ts) else -1
                with torch.no_grad():
                    noise_pred = self._noise_pred_net(
                        x,
                        torch.tensor([t_cur], device=self._device, dtype=torch.long),
                        obs_cond,
                    )
                x = self._scheduler.step(noise_pred, t_cur, x, t_prev)

            # x: (1, action_dim, pred_horizon). robomimic slices
            # [obs_horizon-1 : obs_horizon-1+action_horizon] out of the
            # predicted trajectory (not index 0) before queuing it.
            start = self._obs_horizon - 1
            end = start + self._action_horizon
            chunk = x[0, :, start:end] * self._scale.unsqueeze(-1) + self._offset.unsqueeze(-1)
            for j in range(chunk.shape[1]):
                self._action_queue.append(chunk[:, j].cpu().numpy())

        return self._action_queue.popleft()

    def reset(self) -> None:
        self._obs_buf.clear()
        self._action_queue.clear()


# ============================================================================
# Normalization helper
# ============================================================================

def _load_normalization(
    ckpt: dict, action_dim: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load scale/offset from ckpt, defaulting to identity if absent."""
    stats  = ckpt.get("action_normalization_stats", {}).get("actions", {})
    ones   = np.ones(action_dim,  dtype=np.float32)
    zeros  = np.zeros(action_dim, dtype=np.float32)
    # reshape(-1): robomimic sometimes stores these with a leading (1, D)
    # batch dim instead of flat (D,) — flatten so callers get a predictable
    # 1-D tensor regardless of which shape the checkpoint used.
    scale  = torch.tensor(stats.get("scale",  ones),  dtype=torch.float32).reshape(-1).to(device)
    offset = torch.tensor(stats.get("offset", zeros), dtype=torch.float32).reshape(-1).to(device)
    return scale, offset


def _load_obs_normalization(ckpt: dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load per-key (offset, scale) obs stats from ckpt's obs_normalization_stats,
    if present. Observations are normalized at inference as (raw - offset) / scale,
    matching the mean/std standardization robomimic applies when hdf5_normalize_obs
    is enabled during training. Keys absent from the checkpoint are left unnormalized."""
    stats = ckpt.get("obs_normalization_stats") or {}
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key, s in stats.items():
        if isinstance(s, dict) and "offset" in s and "scale" in s:
            offset = np.array(s["offset"], dtype=np.float32).reshape(-1)
            scale = np.array(s["scale"], dtype=np.float32).reshape(-1)
            out[key] = (offset, scale)
    return out


def _discover_mlp_layer_dims(nets: Dict[str, torch.Tensor], prefix: str) -> List[int]:
    """Output width of each Linear layer under `prefix` (robomimic's MLP is an
    nn.Sequential alternating Linear/ReLU, so Linear layers sit at even indices),
    in order, read directly from the checkpoint — no assumption about depth or
    whether widths are uniform across layers."""
    dims = []
    i = 0
    while f"{prefix}.{i}.weight" in nets:
        dims.append(nets[f"{prefix}.{i}.weight"].shape[0])
        i += 2
    if not dims:
        raise ValueError(f"No MLP layers found under checkpoint prefix '{prefix}'")
    return dims


# ============================================================================
# Detection & factory
# ============================================================================

def detect_model_type(checkpoint_path: str) -> str:
    """
    Inspect the checkpoint and return one of:
      bc_mlp | bc_rnn | bc_gmm | diffusion | act | torch_script
    """
    # 1. TorchScript (.pt / .pth saved with torch.jit.save)
    try:
        torch.jit.load(checkpoint_path, map_location="cpu")
        logger.info("[detect] TorchScript file detected.")
        return "torch_script"
    except Exception:
        pass

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 2. Explicit tag embedded in checkpoint
    if "model_type" in ckpt:
        detected = ckpt["model_type"]
        logger.info(f"[detect] Explicit model_type tag: '{detected}'")
        return detected

    # 3. Robomimic-style  (ckpt["model"]["nets"] exists)
    nets = ckpt.get("model", {}).get("nets", {})
    if nets:
        keys = set(nets.keys())
        if any("rnn" in k for k in keys):
            logger.info("[detect] Robomimic BC_RNN checkpoint.")
            return "bc_rnn"
        if any("transformer" in k or "embed_encoder" in k for k in keys):
            raise ValueError(
                "Detected a transformer-encoder-based robomimic checkpoint "
                "(BC-Transformer / BC-Transformer-GMM) — this is not "
                "supported by --backend custom, which only implements a "
                "plain MLP encoder for the 'bc_gmm' model type. Re-run with "
                "--backend robomimic instead, which loads this checkpoint "
                "through robomimic's own algo_factory and selects the "
                "correct architecture automatically."
            )
        if any(k.endswith("decoder.nets.mean.weight") for k in keys):
            logger.info("[detect] Robomimic BC_GMM checkpoint.")
            return "bc_gmm"
        if any(k.endswith("decoder.nets.action.weight") for k in keys):
            logger.info("[detect] Robomimic BC_MLP checkpoint.")
            return "bc_mlp"

    # 4. Diffusion Policy
    flat = _flatten_keys(ckpt)
    if any("noise_pred_net" in k for k in flat):
        logger.info("[detect] Diffusion Policy checkpoint.")
        return "diffusion"

    # 5. ACT
    if "model_state_dict" in ckpt:
        if any("transformer" in k for k in ckpt["model_state_dict"]):
            logger.info("[detect] ACT checkpoint.")
            return "act"

    raise ValueError(
        f"Cannot auto-detect model type for '{checkpoint_path}'.\n"
        "Supported types: bc_mlp, bc_rnn, bc_gmm, diffusion, act, torch_script.\n"
        "Options:\n"
        "  • Pass --model-type <type> to override detection.\n"
        "  • Add 'model_type': '<type>' key to your checkpoint dict and re-save."
    )


def _flatten_keys(obj, prefix: str = "") -> List[str]:
    """Recursively collect all string keys in a nested dict."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else k
            out.append(full)
            out.extend(_flatten_keys(v, full))
    return out


def load_policy(
    checkpoint_path: str,
    device: torch.device,
    obs_keys: Optional[List[str]],
    model_type: str,
    action_horizon: Optional[int] = None,
) -> BasePolicy:
    """Instantiate the correct BasePolicy subclass for model_type."""
    if action_horizon is not None and model_type != "diffusion":
        logger.warning(
            f"[load_policy] action_horizon override is only used by model_type='diffusion' "
            f"(got model_type='{model_type}') — ignoring."
        )

    if model_type == "torch_script":
        return TorchScriptPolicy(checkpoint_path, device, obs_keys)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    builders = {
        "bc_mlp":    lambda: RobomimicBCMLPPolicy(ckpt, device, obs_keys),
        "bc_rnn":    lambda: RobomimicBCRNNPolicy(ckpt, device, obs_keys),
        "bc_gmm":    lambda: RobomimicBCGMMPolicy(ckpt, device, obs_keys),
        "diffusion": lambda: _RobomimicDiffusionPolicy(ckpt, device, obs_keys, action_horizon),
        "act":       lambda: ACTPolicy(ckpt, device, obs_keys),
    }

    if model_type not in builders:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {sorted(builders) + ['torch_script']}"
        )

    return builders[model_type]()


# ============================================================================
# ModelRunner — public facade (same interface as before)
# ============================================================================

class ModelRunner:
    """
    Loads any supported policy and exposes a uniform inference interface.

    Usage:
        runner = ModelRunner("model.pth")                    # auto-detect type
        runner = ModelRunner("model.pth", model_type="bc_rnn")  # explicit
        runner = ModelRunner("model.pt",  model_type="torch_script",
                             obs_keys=["robot0_eef_pos", "robot0_eef_quat"])

        action = runner.get_action(obs_dict)  # same as before
        runner.reset()                        # reset RNN state (no-op for MLP/GMM)

    obs_keys controls which keys are extracted from obs_dict and in what order.
    If omitted, defaults to BasePolicy.DEFAULT_OBS_KEYS.
    TorchScript models require obs_keys to be set explicitly.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        obs_keys: Optional[List[str]] = None,
        model_type: str = "auto",
        action_horizon: Optional[int] = None,
    ):
        self._device = torch.device(device)

        detected = (
            model_type
            if model_type != "auto"
            else detect_model_type(checkpoint_path)
        )
        logger.info(f"[ModelRunner] type={detected}  device={device}  checkpoint={checkpoint_path}")
        self._policy = load_policy(checkpoint_path, self._device, obs_keys, detected, action_horizon)
        logger.info(f"[ModelRunner] obs_keys: {self._policy.obs_keys}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Run one forward pass. Returns action as np.ndarray."""
        return self._policy.get_action(obs_dict)

    def reset(self) -> None:
        """Reset stateful policy components. Call between episodes for RNN/ACT policies."""
        self._policy.reset()

    @property
    def obs_keys(self) -> List[str]:
        return self._policy.obs_keys
