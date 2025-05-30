import random
import numpy as np
import torch
from torch import nn

from .helpers import (
    cosine_beta_schedule,
    extract,
    condition_projection,
    Losses,
    compute_mask,
)


class GaussianDiffusion(nn.Module):
    """
    Implementation of a Gaussian Diffusion model for sequence generation tasks.
    Supports both standard diffusion sampling and DDIM (Denoising Diffusion Implicit Models) for faster inference.
    """
    def __init__(self, args, model, ddim_discr_method='uniform'):
        """
        Initialize the Gaussian Diffusion model.
        
        Args:
            args: Configuration arguments containing model parameters
            model: The denoising network model (typically a UNet)
            ddim_discr_method: Method for DDIM timestep discretization ('uniform' or 'quad')
        """
        super().__init__()
        # Set model dimensions and parameters
        self.horizon = args.horizon                # Sequence length
        self.observation_dim = args.observation_dim # Dimension of observation vectors
        self.action_dim = args.action_dim          # Dimension of action vectors
        self.class_dim = args.class_dim            # Dimension for class embeddings
        self.model = model                         # Denoising network (e.g., UNet)
        self.weight = args.weight                  # Weight for loss function
        self.ifMask = args.ifMask                  # Whether to use masking
        self.kind = args.kind                      # Type of training approach
        self.mask_loss = args.mask_loss            # Mask loss specification
        self.mask_iteration = args.mask_iteration  # How masking is applied during iterations
        self.dataset = args.dataset                # Dataset being used

        # Diffusion process parameters
        self.n_timesteps = args.n_diffusion_steps  # Number of diffusion steps (default=200)
        self.clip_denoised = args.clip_denoised    # Whether to clip denoised output to [-1,1]
        self.eta = 0.0                             # Eta parameter for DDIM sampling noise control
        self.random_ratio = 1.0                    # Scaling factor for random noise

        # Loss function parameters
        self.l_order = args.l_order                # Order loss weight
        self.l_pos = args.l_pos                    # Position loss weight 
        self.l_perm = args.l_perm                  # Permutation loss weight

        # Calculate diffusion schedule using cosine function
        betas = cosine_beta_schedule(self.n_timesteps)  # Noise schedule
        alphas = 1. - betas                              # Signal preservation rate
        alphas_cumprod = torch.cumprod(alphas, dim=0)    # Cumulative product of alphas
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])  # Shifted alphas_cumprod

        # ---------------------------ddim (Denoising Diffusion Implicit Models)--------------------------------
        # DDIM sampling uses fewer timesteps for faster generation
        ddim_timesteps = 10  # Number of DDIM sampling steps

        # Select timesteps based on discretization method
        if ddim_discr_method == 'uniform':
            # Uniformly space the timesteps
            c = self.n_timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.n_timesteps, c)))
        elif ddim_discr_method == 'quad':
            # Quadratic spacing of timesteps (focuses more on earlier steps)
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.n_timesteps), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise RuntimeError("Unknown DDIM discretization method")

        self.ddim_timesteps = ddim_timesteps
        self.ddim_timestep_seq = ddim_timestep_seq
        # ----------------------------------------------------------------

        # Register buffers for various terms used in the diffusion process
        # These are pre-computed values for efficient sampling and training
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod - 1))

        # Posterior variance (q(x_{t-1} | x_t, x_0)) calculation
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # Loss configuration
        self.loss_type = args.loss_type    # Type of loss function to use
        self.mask_scale = args.mask_scale  # Scaling factor for masked regions
        self.loss_fn = Losses[self.loss_type](
            self.action_dim, self.class_dim, self.weight, self.mask_scale)

    # ------------------------------------------ sampling ------------------------------------------#

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the mean and variance of the posterior distribution q(x_{t-1} | x_t, x_0).
        
        Args:
            x_start: The predicted x_0
            x_t: The current noisy sample x_t
            t: Current timestep
            
        Returns:
            posterior_mean: Mean of q(x_{t-1} | x_t, x_0)
            posterior_variance: Variance of q(x_{t-1} | x_t, x_0)
            posterior_log_variance_clipped: Log variance clipped for numerical stability
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, mask=None):
        """
        Calculate the mean and variance of p(x_{t-1} | x_t) using the model prediction.
        
        Args:
            x: Current noisy sample x_t
            cond: Conditioning information
            t: Current timestep
            mask: Optional mask to apply to the reconstruction
            
        Returns:
            model_mean: Predicted mean for the denoising step
            posterior_variance: Variance for the denoising step
            posterior_log_variance: Log variance for the denoising step
        """
        x_recon = self.model(x, t)  # Reconstruct x_0 from the model

        if mask is not None:
            x_recon = x_recon * mask  # Apply mask if provided

        # Clip the denoised output to [-1, 1] if specified
        if self.clip_denoised:
            x_recon.clamp(-1., 1.)
        else:
            raise RuntimeError("Clipping denoised output is disabled")

        # Calculate posterior distribution parameters
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        Predict the noise component (epsilon) from x_t and the predicted x_0.
        
        Args:
            x_t: Current noisy sample
            t: Current timestep
            pred_xstart: Predicted x_0
            
        Returns:
            Predicted noise component
        """
        return \
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) \
            / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    @torch.no_grad()
    def p_sample_ddim(self, x, cond, t, t_prev, if_prev=False, if_visualize=False):
        """
        Perform a single DDIM sampling step.
        
        Args:
            x: Current sample x_t
            cond: Conditioning information
            t: Current timestep
            t_prev: Previous timestep to jump to
            if_prev: Whether to use alphas_cumprod_prev
            if_visualize: Whether to collect visualization data
            
        Returns:
            Sampled x_{t-1}
        """
        b, *_, device = *x.shape, x.device
        
        # Reconstruct x_0 from the model
        x_recon = self.model(x, t, cond['observation'], if_visualize)

        # Clip the denoised output if specified
        if self.clip_denoised:
            x_recon.clamp(-1., 1.)
        else:
            raise RuntimeError("Clipping denoised output is disabled")

        # Extract noise component
        eps = self._predict_eps_from_xstart(x, t, x_recon)
        
        # Calculate alpha values for current and target timesteps
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        if if_prev:
            alpha_bar_prev = extract(self.alphas_cumprod_prev, t_prev, x.shape)
        else:
            alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x.shape)
            
        # Calculate sigma for DDIM
        sigma = (
            self.eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Sample random noise
        noise = torch.randn_like(x) * self.random_ratio
        
        # Calculate mean prediction
        mean_pred = (
            x_recon * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        # Create mask to handle the t=0 case
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        
        # Return x_{t-1}
        return mean_pred + nonzero_mask * sigma * noise

    @torch.no_grad()
    def p_sample(self, x, cond, t, mask=None):
        """
        Perform a single step of standard diffusion sampling (DDPM).
        
        Args:
            x: Current sample x_t
            cond: Conditioning information
            t: Current timestep
            mask: Optional mask to apply
            
        Returns:
            Sampled x_{t-1}
        """
        b, *_, device = *x.shape, x.device

        # Get model predictions
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, mask=mask)
            
        # Sample random noise
        noise = torch.randn_like(x) * self.random_ratio

        # Create mask to handle the t=0 case
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        
        # Return x_{t-1}
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, if_jump, if_visualize=False):
        """
        Run the full sampling loop to generate sequences, supporting both DDPM and DDIM.
        
        Args:
            cond: Conditioning information
            if_jump: Whether to use DDIM (True) or DDPM (False)
            if_visualize: Whether to collect visualization data
            
        Returns:
            Generated sequence
        """
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = self.horizon
        shape = (batch_size, horizon, self.class_dim +
                 self.action_dim + self.observation_dim)

        # Initialize x_t with random noise
        x = torch.randn(shape, device=device) * self.random_ratio  # Initialize xt for Noise and diffusion
        # x = torch.zeros(shape, device=device)   # for Deterministic

        # Apply conditioning and masking if needed
        if self.ifMask:
            x = condition_projection(x, cond, self.action_dim, self.class_dim)
            mask = compute_mask(x, self.class_dim,
                                self.action_dim, self.horizon, self.dataset)
            x = x * mask
        else:
            x = condition_projection(x, cond, self.action_dim, self.class_dim)
        
        '''
        The if-else below is for diffusion, should be removed for Noise and Deterministic
        '''
        if not if_jump:  # DDPM - standard diffusion sampling
            for i in reversed(range(0, self.n_timesteps)):
                timesteps = torch.full(
                    (batch_size,), i, device=device, dtype=torch.long)

                x = self.p_sample(x, cond, timesteps, mask=mask)

                x = condition_projection(
                    x, cond, self.action_dim, self.class_dim)

        else:  # DDIM - faster sampling with fewer timesteps
            for i in reversed(range(0, self.ddim_timesteps)):
                timesteps = torch.full(
                    (batch_size,), self.ddim_timestep_seq[i], device=device, dtype=torch.long)
                if i == 0:
                    timesteps_prev = torch.full(
                        (batch_size,), 0, device=device, dtype=torch.long)
                    x = self.p_sample_ddim(
                        x, cond, timesteps, timesteps_prev, True, if_visualize)
                else:
                    timesteps_prev = torch.full(
                        (batch_size,), self.ddim_timestep_seq[i-1], device=device, dtype=torch.long)
                    x = self.p_sample_ddim(
                        x, cond, timesteps, timesteps_prev, False, if_visualize)
                x = condition_projection(
                    x, cond, self.action_dim, self.class_dim)
                if self.mask_iteration == "add":
                    x = x * mask

        '''
        The two lines below are for Noise and Deterministic
        '''
        # x = self.model(x, None)
        # x = condition_projection(x, cond, self.action_dim, self.class_dim)

        return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from q(x_t | x_0) - the forward diffusion process.
        
        Args:
            x_start: The initial clean data x_0
            t: Timestep to sample at
            noise: Noise to use (will generate if not provided)
            
        Returns:
            A sample x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start) * self.random_ratio

        # Apply the diffusion process to add noise
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        """
        Compute the loss for the denoising process.
        
        Args:
            x_start: The clean data x_0
            cond: Conditioning information
            t: Timestep to evaluate at
            
        Returns:
            Calculated loss value
        """
        # Generate noise for diffusion process
        noise = torch.randn_like(x_start) * self.random_ratio  # for Noise and diffusion
        # noise = torch.zeros_like(x_start)   # for Deterministic
        # x_noisy = noise   # for Noise and Deterministic
        
        mask = None
        if self.ifMask:
            mask = compute_mask(x_start, self.class_dim,
                                self.action_dim, self.horizon, self.dataset)
            x_start = x_start * mask
        
        # Apply forward diffusion to get noisy sample
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Apply conditioning
        x_noisy = condition_projection(
            x_noisy, cond, self.action_dim, self.class_dim)

        # Run model to denoise the sample
        x_recon = self.model(x_noisy, t, cond['observation'])

        # Re-apply conditioning to ensure constraints
        x_recon = condition_projection(
            x_recon, cond, self.action_dim, self.class_dim)

        # Calculate appropriate loss based on configuration
        if self.loss_type == 'Sequence_CE':
            loss = self.loss_fn(x_recon, x_start, self.l_order,
                                self.l_pos, self.l_perm, self.kind)
        elif self.mask_loss == '1':
            loss = self.loss_fn(x_recon, x_start, mask)
        else:
            loss = self.loss_fn(x_recon, x_start)
        
        return loss

    def loss(self, x, cond):
        """
        Calculate the overall loss for a batch.
        
        Args:
            x: Clean data x_0 batch
            cond: Conditioning information
            
        Returns:
            Batch loss
        """
        batch_size = len(x)  # Get the batch size

        self.observation_img = x[:, :, :]

        # Sample random timesteps for each item in batch
        t = torch.randint(0, self.n_timesteps, (batch_size,),
                          device=x.device).long()  # Random timestep for diffusion
        # t = None    # for Noise and Deterministic
        
        return self.p_losses(x, cond, t)

    def forward(self, cond, if_jump=True, if_visualize=False):
        """
        Forward pass - generates a sample given conditioning information.
        
        Args:
            cond: Conditioning information
            if_jump: Whether to use DDIM sampling
            if_visualize: Whether to collect visualization data
            
        Returns:
            Generated sample
        """
        return self.p_sample_loop(cond, True, if_visualize)
