import os
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import torch.utils.data
from einops import rearrange
from PIL import Image
from torchvision import transforms


class PoisonGeneration(object):
    def __init__(
        self,
        target_concept,
        device,
        eps=0.05,
        constraint="linf",
        lpips_lambda=10.0,
        lpips_net="vgg",
        steps=500,
    ):
        self.eps = eps
        self.target_concept = target_concept
        self.device = device
        self.constraint = constraint
        self.lpips_lambda = lpips_lambda
        self.lpips_net = lpips_net
        self.steps = steps
        self.full_sd_model = self.load_model()
        self.transform = self.resizer()
        if self.constraint not in {"linf", "lpips"}:
            raise ValueError("constraint must be either 'linf' or 'lpips'")
        self.lpips_model = self.load_lpips_model() if self.constraint == "lpips" else None

    def resizer(self):
        image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
            ]
        )
        return image_transforms

    def load_model(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            "Manojb/stable-diffusion-2-1-base",
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        pipeline = pipeline.to(self.device)
        return pipeline

    def load_lpips_model(self):
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPS mode requires the `lpips` package in the active environment."
            ) from exc

        lpips_model = lpips.LPIPS(net=self.lpips_net).to(self.device)
        lpips_model.eval()
        for param in lpips_model.parameters():
            param.requires_grad = False
        return lpips_model

    def generate_target(self, prompts):
        torch.manual_seed(123)  # ensuring the target image is consistent across poison set
        with torch.no_grad():
            target_imgs = self.full_sd_model(prompts, guidance_scale=7.5, num_inference_steps=100,
                                             height=512, width=512).images
        target_imgs[0].save("target_%s.png" % prompts)
        return target_imgs[0]

    def get_latent(self, tensor):
        latent_features = self.full_sd_model.vae.encode(tensor).latent_dist.mean
        return latent_features

    def get_lpips(self, source_tensor, adv_tensor):
        return self.lpips_model(adv_tensor.float(), source_tensor.float()).mean()

    def generate_one_linf(self, source_tensor, target_latent):
        modifier = torch.clone(source_tensor) * 0.0

        t_size = self.steps
        max_change = self.eps / 0.5  # scale from 0,1 to -1,1
        step_size = max_change

        for i in range(t_size):
            actual_step_size = step_size - (step_size - step_size / 100) / t_size * i
            modifier.requires_grad_(True)

            adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
            adv_latent = self.get_latent(adv_tensor)

            loss = (adv_latent - target_latent).norm()

            tot_loss = loss.sum()
            grad = torch.autograd.grad(tot_loss, modifier)[0]

            modifier = modifier - torch.sign(grad) * actual_step_size
            modifier = torch.clamp(modifier, -max_change, max_change)
            modifier = modifier.detach()

            if i % 50 == 0:
                print("# Iter: {}	Latent Loss: {:.3f}".format(i, loss.mean().item()))

        final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)
        return tensor2img(final_adv_batch)

    def generate_one_lpips(self, source_tensor, target_latent):
        source_tensor_fp32 = source_tensor.float()
        modifier = torch.zeros_like(source_tensor_fp32)

        t_size = self.steps
        step_size = 0.01

        best_adv = source_tensor_fp32.detach().clone()
        best_latent_loss = float("inf")
        best_violation = float("inf")
        best_in_constraint = False

        for i in range(t_size):
            actual_step_size = step_size - (step_size - step_size / 10) / t_size * i
            modifier.requires_grad_(True)

            adv_tensor = torch.clamp(modifier + source_tensor_fp32, -1, 1)
            adv_latent = self.get_latent(adv_tensor.half())

            latent_loss = (adv_latent - target_latent).norm()
            lpips_loss = self.get_lpips(source_tensor_fp32, adv_tensor)
            lpips_overflow = torch.relu(lpips_loss - self.eps)
            tot_loss = latent_loss.float().sum() + self.lpips_lambda * (lpips_overflow ** 2)
            grad = torch.autograd.grad(tot_loss, modifier)[0]

            grad_scale = grad.abs().mean().clamp_min(1e-8)
            modifier = modifier - grad / grad_scale * actual_step_size
            modifier = modifier.detach()

            cur_latent_loss = latent_loss.mean().item()
            cur_lpips = lpips_loss.item()
            cur_violation = max(cur_lpips - self.eps, 0.0)

            if cur_violation == 0.0:
                if (not best_in_constraint) or cur_latent_loss < best_latent_loss:
                    best_adv = adv_tensor.detach()
                    best_latent_loss = cur_latent_loss
                    best_in_constraint = True
            elif not best_in_constraint:
                if cur_violation < best_violation or (
                    cur_violation == best_violation and cur_latent_loss < best_latent_loss
                ):
                    best_adv = adv_tensor.detach()
                    best_latent_loss = cur_latent_loss
                    best_violation = cur_violation

            if i % 50 == 0:
                print(
                    "# Iter: {}	Latent Loss: {:.3f}	LPIPS: {:.4f}	Budget: {:.4f}".format(
                        i, cur_latent_loss, cur_lpips, self.eps
                    )
                )

        return tensor2img(best_adv)

    def generate_one(self, pil_image, target_concept):

        resized_pil_image = self.transform(pil_image)
        source_tensor = img2tensor(resized_pil_image).to(self.device)

        target_image = self.generate_target("{} image".format(target_concept))
        target_tensor = img2tensor(target_image).to(self.device)

        target_tensor = target_tensor.half()
        source_tensor = source_tensor.half()

        with torch.no_grad():
            target_latent = self.get_latent(target_tensor)

        if self.constraint == "lpips":
            return self.generate_one_lpips(source_tensor, target_latent)
        return self.generate_one_linf(source_tensor, target_latent)

    def generate_all(self, image_paths, target_concept):
        res_imgs = []
        for idx, image_f in enumerate(image_paths):
            cur_img = image_f.convert('RGB')
            perturbed_img = self.generate_one(cur_img, target_concept)
            res_imgs.append(perturbed_img)
        return res_imgs


def img2tensor(cur_img):
    cur_img = cur_img.resize((512, 512), resample=Image.Resampling.BICUBIC)
    cur_img = np.array(cur_img)
    img = (cur_img / 127.5 - 1.0).astype(np.float32)
    img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img).unsqueeze(0)
    return img


def tensor2img(cur_img):
    if len(cur_img) == 512:
        cur_img = cur_img.unsqueeze(0)

    cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
    cur_img = 255. * rearrange(cur_img[0], 'c h w -> h w c').cpu().numpy()
    cur_img = Image.fromarray(cur_img.astype(np.uint8))
    return cur_img
