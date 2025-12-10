import torch
import torch.nn as nn
import torch.optim as optim
import copy
import tqdm
import math

from .SD1_5 import SD1_5
from .SD3 import SD3
from .FLUX import FLUX 

class EMA:
    def __init__(self, beta: float = 0.9999):
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None: return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class LatentDiffusionTrainer:
    def __init__(self, 
                 model_type: str = "SD1.5", # "SD1.5", "SD3", "FLUX"
                 learn_rate: float = 1e-4, 
                 number_of_epochs: int = 10,
                 device: str = "cuda"):
        
        self.model_type = model_type
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.device = device
        
        if self.model_type == "SD1.5":
            self.model = SD1_5(device=self.device)
            self.trainable_module = self.model.unet
        elif self.model_type == "SD3":
            self.model = SD3(device=self.device)
            self.trainable_module = self.model.transformer
        elif self.model_type == "FLUX":
            self.model = FLUX(device=self.device)
            self.trainable_module = self.model.transformer
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        self.optimizer = optim.AdamW(self.trainable_module.parameters(), lr=self.learn_rate)
        self.mse = nn.MSELoss()
        self.scaler = torch.amp.GradScaler("cuda")
        
        self.ema = EMA(beta=0.9999)
        self.ema_model = copy.deepcopy(self.trainable_module).eval().requires_grad_(False)
        
    def fit(self, dataloader: torch.utils.data.DataLoader, verbose: bool = False):
        self.trainable_module.train()
        print(f"Starting training {self.model_type} on {self.device}...")
        
        for epoch in range(self.number_of_epochs):
            pbar = tqdm.tqdm(dataloader)
            epoch_loss = 0
            
            for i, (images, captions) in enumerate(pbar):
                images = images.to(self.device)
                
                latents = self.model.encode_images(images) 
                bs = latents.shape[0]
                
                loss = 0
                self.optimizer.zero_grad()
                
                with torch.amp.autocast("cuda"):
                    
                    if self.model_type == "SD1.5":
                        with torch.no_grad():
                            context = self.model.text_encoder(captions)
                        
                        t = self.model.scheduler.sample_timesteps(bs)
                        x_t, noise = self.model.scheduler.noise_images(latents, t)
                        
                        predicted_noise = self.model.unet(x_t, t, context)
                        loss = self.mse(noise, predicted_noise)
                        
                    elif self.model_type in ["SD3", "FLUX"]:
                        with torch.no_grad():
                            if self.model_type == "SD3":
                                y, context = self.model.text_encoder(captions)
                            else: 
                                y, context = self.model.text_encoder(captions)

                        if self.model_type == "FLUX":
                            target_latents = latents.flatten(2).transpose(1, 2) 
                        else:
                            target_latents = latents

                        noise = torch.randn_like(target_latents)
                        
                        t = torch.rand((bs,), device=self.device)
                        t_expand = t.view(bs, 1, 1) if self.model_type == "FLUX" else t.view(bs, 1, 1, 1)

                        x_t = (1 - t_expand) * target_latents + t_expand * noise
                        target_v = noise - target_latents

                        if self.model_type == "SD3":
                            pred_v = self.model.transformer(x_t, t, y, context)
                            
                        elif self.model_type == "FLUX":
                            guidance = torch.ones((bs,), device=self.device) 
                            
                            pred_v = self.model.transformer(
                                img=x_t, 
                                t=t, 
                                guidance=guidance, 
                                txt_pooled=y, 
                                txt_t5=context
                            )

                        loss = self.mse(target_v, pred_v)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.ema.step_ema(self.ema_model, self.trainable_module)
                
                loss_val = loss.item()
                epoch_loss += loss_val
                pbar.set_postfix(Type=self.model_type, Epoch=f"{epoch+1}/{self.number_of_epochs}", MSE=f"{loss_val:.4f}")
            
            if verbose:
                print(f"Epoch {epoch+1} finished | Avg Loss: {epoch_loss/len(dataloader):.5f}")

    def sample(self, prompt: str, steps: int = None, guidance_scale: float = None):
        print(f"Sampling prompt: '{prompt}' with {self.model_type}")
        
        if steps is None:
            if self.model_type == "SD1.5": steps = 50
            elif self.model_type == "SD3": steps = 28
            elif self.model_type == "FLUX": steps = 20
        
        if guidance_scale is None:
            if self.model_type == "SD1.5": guidance_scale = 7.5
            elif self.model_type == "SD3": guidance_scale = 7.0
            elif self.model_type == "FLUX": guidance_scale = 3.5

        if self.model_type == "SD1.5":
            return self.model.generate(prompt, steps=steps, cfg_scale=guidance_scale, unet_override=self.ema_model)
        elif self.model_type == "SD3":
            return self.model.generate(prompt, steps=steps, cfg_scale=guidance_scale, transformer_override=self.ema_model)
        elif self.model_type == "FLUX":
            return self.model.generate(prompt, steps=steps, guidance_scale=guidance_scale, transformer_override=self.ema_model)