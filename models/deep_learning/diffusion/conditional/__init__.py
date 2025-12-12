import torch
import torch.nn as nn
import torch.optim as optim
import copy
import tqdm

from .SD1_5 import SD1_5
from .Kandinsky import Kandinsky2_2
from .Hunyuan import HunyuanDiT
from .SD3 import SD3
from .FLUX import FLUX 

class EMA():
    def __init__(self, beta: float = 0.9999) -> None:
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

class LatentDiffusionTrainer():
    def __init__(self, 
                 model_type: str = "SD1.5", # "SD1.5", "SD3", "FLUX"
                 learn_rate: float = 1e-4, 
                 number_of_epochs: int = 10,
                 device: str = "cuda") -> None:
        """
        Latent Diffusion Model Trainer for various architectures
        
        Parameters:
            model_type: Type of the diffusion model ("SD1.5", "SD3", "FLUX")
            learn_rate: Learning rate for the optimizer
            number_of_epochs: Number of training epochs
            device: Device to run the training on ("cuda" or "cpu")
        """
        self.model_type = model_type
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.device = device
        
        if self.model_type == "SD1.5":
            self.model = SD1_5(device=self.device)
            self.trainable_module = self.model.unet
        elif self.model_type == "HUNYUAN":
            self.model = HunyuanDiT(device=self.device)
            self.trainable_module = self.model
        elif self.model_type == "SD3":
            self.model = SD3(device=self.device)
            self.trainable_module = self.model.transformer
        elif self.model_type == "FLUX":
            self.model = FLUX(device=self.device)
            self.trainable_module = self.model.transformer
        elif self.model_type == "KANDINSKY":
            self.model = Kandinsky2_2(device=self.device)
            self.trainable_module = self.model.unet
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        self.optimizer = optim.AdamW(self.trainable_module.parameters(), lr=self.learn_rate)
        self.mse = nn.MSELoss()
        
        self.ema = EMA(beta=0.9999)
        self.ema_model = copy.deepcopy(self.trainable_module).eval().requires_grad_(False)
        
    def fit(self, 
            dataloader: torch.utils.data.DataLoader, 
            verbose: bool = False) -> None:
        self.trainable_module.train()
        print(f"Starting training {self.model_type} on {self.device}...")
        
        for epoch in range(self.number_of_epochs):
            pbar = tqdm.tqdm(dataloader)
            epoch_loss = 0
            
            for _, (images, captions) in enumerate(pbar):
                images = images.to(self.device)
                
                latents = self.model.encode_images(images) 
                bs = latents.shape[0]
                
                loss = 0
                self.optimizer.zero_grad()
                
                # Noise prediction target
                if self.model_type in ["SD1.5", "HUNYUAN", "KANDINSKY"]:
                    with torch.no_grad():
                        if self.model_type == "KANDINSKY":
                            context = self.model.embedder.get_text_embeds(captions)
                            image_embs = self.model.embedder.get_image_embeds(images)
                        else:
                            context = self.model.text_encoder(captions)
                    
                    t = self.model.scheduler.sample_timesteps(bs)
                    x_t, noise = self.model.scheduler.noise_images(latents, t)
                    
                    if self.model_type == "SD1.5":
                        predicted_noise = self.model.unet(x_t, t, context)
                    elif self.model_type == "HUNYUAN":
                        predicted_noise = self.model(x_t, t, context)
                    elif self.model_type == "KANDINSKY":
                        predicted_noise = self.model.unet(x_t, t, context, image_embs)
                        
                    loss = self.mse(noise, predicted_noise)
                
                # Velocity prediction target
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
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_module.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.ema.step_ema(self.ema_model, self.trainable_module)
                
                loss_val = loss.item()
                epoch_loss += loss_val
                pbar.set_postfix(Type=self.model_type, Epoch=f"{epoch + 1}/{self.number_of_epochs}", MSE=f"{loss_val:.4f}")
            
            if verbose:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] | Avg Loss: {epoch_loss / len(dataloader):.5f}")

    def sample(self, prompt: str, steps: int = None, guidance_scale: float = None):
        print(f"Sampling prompt: '{prompt}' with {self.model_type}")
        
        if steps is None:
            steps = 28 if self.model_type in ["SD3", "FLUX"] else 50
        
        if guidance_scale is None:
            guidance_scale = 3.5 if self.model_type == "FLUX" else 7.5

        if self.model_type in ["SD1.5", "KANDINSKY"]:
            return self.model.generate(
                prompt=prompt, 
                steps=steps, 
                cfg_scale=guidance_scale, 
                unet_override=self.ema_model
            )
        else:
            return self.model.generate(
                prompt=prompt, 
                steps=steps, 
                cfg_scale=guidance_scale, 
                transformer_override=self.ema_model
            )