import copy
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from .DDIM import *

class EMA():
    def __init__(self, beta: float = 0.995) -> None:
        """
        Exponential Moving Average (EMA) for model parameters
        
        Parameters:
            beta: The decay rate for the moving average
        """
        self.beta = beta
        self.step = 0

    def update_model_average(self, 
                             ma_model: nn.Module, 
                             current_model: nn.Module) -> None:
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, 
                       old: torch.Tensor, 
                       new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, 
                 ema_model: nn.Module, 
                 model: nn.Module, 
                 step_start_ema: int = 2000) -> None:
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, 
                         ema_model: nn.Module, 
                         model: nn.Module) -> None:
        ema_model.load_state_dict(model.state_dict())

class DDIM():
    def __init__(self,
                 learn_rate: float,
                 number_of_epochs: int,
                 device: str = "cuda") -> None:
        
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.device = device

    def init_network(self) -> None:
        """
        Initialize the diffusion model, optimizer, and loss function
        """
        self.model = UNet(device=self.device).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learn_rate)
        self.mse = nn.MSELoss()
        self.diffusion = DiffusionUtils(device=self.device, schedule_name="cosine") 
        self.scaler = torch.amp.GradScaler("cuda")

        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

    def fit(self, 
            dataloader: torch.utils.data.DataLoader,
            verbose: bool = False) -> None:
        """
        Train the DDIM model using the provided DataLoader

        Parameters:
            dataloader: DataLoader for training data
            verbose: If True, print training progress
        """
        self.init_network()
        for epoch in range(self.number_of_epochs):
            pbar = tqdm(dataloader)
            epoch_loss = 0
            
            for _, (images) in enumerate(pbar):
                images = images.to(self.device)
                
                t = self.diffusion.sample_timesteps(images.shape[0])
                x_t, noise = self.diffusion.noise_images(images, t)
                
                with torch.amp.autocast('cuda'):
                    predicted_noise = self.model(x_t, t)
                    loss = self.mse(noise, predicted_noise)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.ema.step_ema(self.ema_model, self.model)
                
                loss_value = loss.item()
                epoch_loss += loss_value
                pbar.set_postfix(MSE=loss_value)
            
            avg_loss = epoch_loss / len(dataloader)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] | Avg Loss: {avg_loss:.5f}")
    
    def sample(self,
               n: int,
               ddim_timesteps: int = 50, 
               eta: float = 0.0) -> torch.Tensor:
        """
        Generate samples from the trained DDIM model

        Parameters:
            n: Number of samples to generate
            ddim_timesteps: Number of DDIM timesteps
            eta: Controls the scale of noise added during sampling
        
        Returns:
            samples: Generated samples as a tensor
        """
        self.ema_model.eval()
        with torch.no_grad():
            samples = self.diffusion.sample_ddim(model=self.ema_model, 
                                                 n=n,
                                                 ddim_timesteps=ddim_timesteps,
                                                 eta=eta)
        return samples

    def __str__(self) -> str:
        return "Denoising Diffusion Implicit Models (DDIM)"