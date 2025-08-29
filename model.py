import torch
from torch import nn
from tqdm.auto import tqdm
from dit import DiT
import torch.nn.functional as F
from einops import rearrange
from functools import partial

def stopgrad(x):
    return x.detach()

def sg_lambda(x, lambda_val):
    return lambda_val * x + (1 - lambda_val) * stopgrad(x)

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Power p = 1 - gamma
    So: gamma=0.5 → p=0.5 (Pseudo-Huber-like)
        gamma=0.0 → p=1.0 (Best according to paper)
        gamma=1.0 → p=0.0 (Standard L2)
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()

class MeanFlow(nn.Module):
    def __init__(
        self,
        device="cuda",
        channels=3,
        image_size=32,
        num_classes=10,
        cfg_drop_prob=0.1,
        lambda_mode="curriculum",
    ):
        super().__init__()
        self.device = device
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None
        self.class_dropout_prob = cfg_drop_prob
        self.jvp_fn = torch.autograd.functional.jvp
        self.create_graph = True
        self.create_graph_threshold = 0.01

    def sample_t_r(self, b, device):
        # Logit-normal sampling
        mu, sigma = -0.4, 1.0
        normal_samples = torch.randn(b, 2, device=device) * sigma + mu
        samples = torch.sigmoid(normal_samples)
        
        # Enforce t > r
        t = torch.max(samples, dim=1)[0]
        r = torch.min(samples, dim=1)[0]
        
        # Set some r = t (e.g., 50%)
        flow_ratio = 0.5
        mask = torch.rand(b, device=device) < flow_ratio
        r[mask] = t[mask]
        
        return t, r

    def get_lambda(self, step=None):
        """
        Get the current lambda value based on the mode and step
        """
        if self.lambda_mode == "fixed":
            return self.fixed_lambda
        elif self.lambda_mode == "curriculum":
            # Curriculum scheduling: lambda(t) = min(1, t/T_warmup)
            if step is None:
                step = self.current_step
            return min(1.0, step / self.warmup_steps)
        else:
            raise ValueError(f"Unknown lambda mode: {self.lambda_mode}")
    
    def loss(self, model, x, z, y, cfg_scale=5.0, step=None):
        """I used forward directly instead of via sampler
        L_λ = E[||u_θ(x_t, r, t) + (t-r)·SG_λ[∂_t u_θ + ∇_x u_θ · (x1-x0)/(t-r)] - (x1-x0)/(t-r)||²]

        """
        device = x.device
        b = x.shape[0]
        t, r = self.sample_t_r(b, device)
        lambda_val = self.get_lambda(step)

        t_hat = rearrange(t, 'b -> b 1 1 1').detach().clone()
        r_hat = rearrange(r, 'b -> b 1 1 1').detach().clone()

        x_t = (1-t_hat) * z + t_hat * x
        v = x - z

        if cfg_scale > 1.0:
            y_uncond_all = torch.full_like(y, self.num_classes)
            drop_mask = torch.rand(b, device=device) < self.class_dropout_prob
            y_input = y.clone()
            y_input[drop_mask] = self.num_classes  # Use num_classes as null token

            # v_cfg(z_t, t | c) = ω × v(z_t, t | c) + (1 - ω) × v(z_t, t)
            # => v_cfg(z_t, t | c) = ω × v(z_t, t | c) + (1 - ω) × u_cfg(z_t, t, t) note: u_cfg(z_t, t, t) = v(z_t, t)
            with torch.no_grad():
                u_at_t = model(x_t, t, t, y_uncond_all)
            v_tilde = cfg_scale * v + (1 - cfg_scale) * u_at_t
        else:
            v_tilde = v
            y_input = y

        create_graph = lambda_val > self.create_graph_threshold
        if lambda_val > 0:
            model_partial = partial(model, y=y_input)
            jvp_args = (
                lambda x_t, r, t: model_partial(x_t, r, t),
                (x_t, r, t),
                (v_tilde,torch.zeros_like(r),  torch.ones_like(t)),
            )

            if self.create_graph:
                u_pred, dudt = self.jvp_fn(*jvp_args, create_graph=True)
            else:
                u_pred, dudt = self.jvp_fn(*jvp_args)

            modulated_dudt = sg_lambda(dudt, lambda_val)
            u_tgt = v_tilde - (t_hat - r_hat) * modulated_dudt
        else:
            u_pred = model(x_t, r, t, y_input)
            
            with torch.no_grad():
                u_tgt = v_tilde
        

        u_tgt = v_tilde - (t_hat-r_hat) * dudt

        error = u_pred - stopgrad(u_tgt)

        loss = adaptive_l2_loss(error)

        return loss, lambda_val
    
    @torch.no_grad()
    def sample(self, model, batch_size=None, class_labels=None):
        if class_labels is not None:
            # Use provided class labels
            batch_size = class_labels.shape[0]
            c = class_labels.to(self.device)
        elif self.use_cond and batch_size is not None:
            # Generate random class labels
            c = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        elif batch_size is not None:
            # No conditioning
            c = None
        else:
            raise ValueError("Either batch_size or class_labels must be provided")
        
        print('class labels: ', c)
        
        z = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)
        r = torch.zeros(batch_size, device=self.device)
        t = torch.ones(batch_size, device=self.device)
        u = model(z, r, t, c)

        x = z + u
        return x
        
    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, sample_steps=5, device='cuda'):
        """Sample n_per_class images for each class."""
        if not self.use_cond:
            raise ValueError("Cannot sample each class when num_classes is None")
        
        c = torch.arange(self.num_classes, device=self.device).repeat(n_per_class)
        z = torch.randn(self.num_classes * n_per_class, self.channels, self.image_size, self.image_size, device=self.device)
        t_vals = torch.linspace(0.0, 1.0, sample_steps + 1, device=device)

        # print(t_vals)

        for i in range(sample_steps):
            r = torch.full((z.size(0),), t_vals[i], device=device)
            t = torch.full((z.size(0),), t_vals[i + 1], device=device)

            # print(f"t: {t[0].item():.4f};  r: {r[0].item():.4f}")
            r_hat = rearrange(r, "b -> b 1 1 1").detach().clone()
            t_hat = rearrange(t, "b -> b 1 1 1").detach().clone()
            

            v = model(z, r, t, c)
            z = z + (t_hat-r_hat) * v

        return z
