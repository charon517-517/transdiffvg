import torch.nn as nn
from diffusers.schedulers import DDIMScheduler
from models.SD_predict_noise_pipeline import SDPredictNoisePipeline

class UnetFeatExtractor(nn.Module):
    def __init__(self, model_path='/home/wangsai/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2/snapshots/1e128c8891e52218b74cde8f26dbfc701cb99d79', hidden_dim=256) -> None:
        super().__init__()
        self.ddim_scheduler = DDIMScheduler.from_pretrained(model_path, subfolder='scheduler')
        self.noise_generator = SDPredictNoisePipeline.from_pretrained(model_path, scheduler=self.ddim_scheduler)
        
    
    def forward(self, inputs, prompt, eta=0.0, num_inference_steps=1, output_type="latent", device=None):
        num_images_per_prompt = inputs.shape[0]
        self.noise_generator = self.noise_generator.to(device)
        predicted_noise, costs = self.noise_generator(inputs, prompt=prompt, eta=eta, num_images_per_prompt=1, num_inference_steps=num_images_per_prompt, output_type="latent")
        return predicted_noise