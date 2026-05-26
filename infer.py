import torch
from PIL import Image
from typing import Union, Optional
from tqdm import tqdm
import os
import random
import numpy as np
import argparse

from diffsynth.pipelines.qwen_image import (
    QwenImagePipeline as _BaseQwenImagePipeline,
    QwenImageUnit_PromptEmbedder as _BasePromptEmbedder,
    QwenImageUnit_ShapeChecker,
    QwenImageUnit_NoiseInitializer,
    QwenImageUnit_InputImageEmbedder,
    QwenImageUnit_Inpaint,
    QwenImageUnit_EditImageEmbedder,
    QwenImageUnit_LayerInputImageEmbedder,
    QwenImageUnit_ContextImageEmbedder,
    QwenImageUnit_EntityControl,
    QwenImageUnit_BlockwiseControlNet,
    QwenImageBlockwiseMultiControlNet,
    model_fn_qwen_image,
)
from diffsynth.core import ModelConfig
from diffsynth.core.device.npu_compatible_device import get_device_type
from diffsynth.diffusion.base_pipeline import PipelineUnit, ControlNetInput
from diffsynth import load_state_dict
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────
system_prompt = (
    "You are an expert in image quality and aesthetic evaluation. "
    "Your task is to perform a comprehensive, multi-level analysis of the given image. "
    "For every judgment, follow a detailed, step-by-step chain-of-thought reasoning. "
    "Ensure your analysis is structured, logical, and fully justified using professional terminology and clear reasoning. "
    "After the evaluation, identify critical issues that should be addressed to improve the image appeal."
)

Quality = (
    "Step 1: Image Quality / Degradations Analysis. "
    "Note: These attributes negatively affect image perception and are mostly interference elements that need removal. "
    "1. Blur / Clarity (motion blur, defocus, etc.) "
    "2. Noise (digital noise, compression artifacts, etc.) "
    "3. Distortion and Artifacts (stretching, moiré, lens distortion, etc.) "
    "4. Visual Interference (glare, shadow occlusion, weather effects, etc.)"
)

Color = (
    "Step 2: Color Performance & Lighting Analysis. "
    "Note: These are basic low-level visual attributes that determine quantifiable visual information. "
    "1. Saturation "
    "2. White balance / Color temperature "
    "3. Contrast "
    "4. Exposure / Light "
    "5. Dynamic Range "
    "6. Shadow quality (some shadows may be intentional artistic design)"
)

Composition = (
    "Step 3: Composition Analysis. "
    "Note: These are technical factors related to static composition and camera parameter settings, crucial for aesthetic evaluation. "
    "1. Composition and Layout "
    "2. Focus "
    "3. Camera viewpoint & horizon "
    "4. Background Blur (Bokeh-like Effect)"
    "5. Photography style (cinematic, portrait, landscape, etc.)"
)

Aesthetic = (
    "Step 4: Aesthetic Impression Analysis. "
    "Note: These high-level attributes reflect perceptual and semantic aesthetic impressions. "
    "1. Tone style (vibrant, warm, cool, filmic, etc.) "
    "2. Creativity "
    "3. Emotional expression and storytelling "
    "4. Semantic richness "
    "5. Environment and background integration "
    "6. Context adaptability "
)

Evaluation = (
    "Step 5: Comprehensive Evaluation and Output Format:\n"
    "Summarize the above analysis and provide concrete improvement suggestions. Clearly explain the reasoning and logic. "
    "Structure the output into numbered sections: "
    "(1) Image Quality / Degradations Analysis + distortion severity, "
    "(2) Color Performance & Lighting Analysis, "
    "(3) Composition Analysis, "
    "(4) Aesthetic Impression Analysis, "
    "(5) Concrete and concise suggestions for image improvement. "
    "Wrap everything in <think>...</think>. "
    "After that, identify the most severe issues and summarize practical, feasible suggestions in <edit_start>...</edit_end>. "
    "Finally, provide the overall quality and aesthetic score (0-100) in <answer>...</answer> without extra words."
)

Suggestions = (
    "VERY IMPORTANT notes: Suggestions for Improvement:\n"
    "Provide concise, practical, and feasible improvement suggestions, organized into numbered categories.\n"
    "Focus only on major quality issues that significantly degrade image perception or aesthetic appeal.\n"
    "Ignore minor deviations or intentional artistic choices.\n"
)

user_prompt = (
    f"{system_prompt}\n"
    f"{Quality}\n"
    f"{Color}\n"
    f"{Composition}\n"
    f"{Aesthetic}\n"
    f"{Evaluation}\n"
    f"{Suggestions}\n"
)

# ─────────────────────────────────────────────
# PromptEmbedder
# ─────────────────────────────────────────────
class QwenImageUnit_PromptEmbedder(_BasePromptEmbedder):

    def __init__(self):
        PipelineUnit.__init__(
            self,
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("edit_image", "output_txt_path"),
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",),
        )

    def _encode_auto_edit(self, pipe, prompt, edit_image: list, output_txt_path=None):
        """
        edit_image: list of PIL.Image
        """
        template = (
            "<|im_start|>system\nYou are a helpful assistant that follows user instructions.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        model_inputs = pipe.processor(
            text=[template],
            images=edit_image,          
            padding=True,
            return_tensors="pt",
        ).to(pipe.device)

        with torch.no_grad():
            outputs = pipe.text_encoder.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                max_new_tokens=1280,
                do_sample=False,
                eos_token_id=pipe.tokenizer.eos_token_id,
                pad_token_id=pipe.tokenizer.pad_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        all_hidden_states = outputs.hidden_states
        hs_prefill = all_hidden_states[0][-1]
        hs_decode = torch.cat([hs[-1] for hs in all_hidden_states[1:]], dim=1)
        hidden_states = torch.cat([hs_prefill, hs_decode], dim=1)

        completion_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(model_inputs.input_ids, outputs.sequences)
        ]
        decoder_outputs = pipe.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )[0]
        print("decoder_outputs:", decoder_outputs)
        pipe._last_decoder_outputs = decoder_outputs

        if output_txt_path is not None:
            with open(output_txt_path, "a", encoding="utf-8") as f:
                f.write(decoder_outputs + "\n\n")

        return [hidden_states.squeeze(0)]

    def process(self, pipe, prompt, edit_image=None, output_txt_path=None) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        if pipe.text_encoder is None:
            return {}

        if isinstance(edit_image, Image.Image):
            edit_image = [edit_image]

        split_hs = self._encode_auto_edit(pipe, prompt, edit_image, output_txt_path)

        attn_masks = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hs]
        max_len = max(e.size(0) for e in split_hs)
        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_len - u.size(0), u.size(1))]) for u in split_hs
        ])
        encoder_attn_mask = torch.stack([
            torch.cat([u, u.new_zeros(max_len - u.size(0))]) for u in attn_masks
        ])
        return {
            "prompt_emb": prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device),
            "prompt_emb_mask": encoder_attn_mask,
        }

# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────
class QwenImagePipeline(_BaseQwenImagePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.units = [
            QwenImageUnit_ShapeChecker(),
            QwenImageUnit_NoiseInitializer(),
            QwenImageUnit_InputImageEmbedder(),
            QwenImageUnit_Inpaint(),
            QwenImageUnit_EditImageEmbedder(),
            QwenImageUnit_LayerInputImageEmbedder(),
            QwenImageUnit_ContextImageEmbedder(),
            QwenImageUnit_PromptEmbedder(),
            QwenImageUnit_EntityControl(),
            QwenImageUnit_BlockwiseControlNet(),
        ]
        self._last_decoder_outputs = None

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        vlm_path: Optional[str] = None,
        vae_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        pipe = QwenImagePipeline(device=device, torch_dtype=torch_dtype)

        effective_configs = list(model_configs)
        if vae_config is not None:
            effective_configs.append(vae_config)

        model_pool = pipe.download_and_load_models(effective_configs, vram_limit)

        pipe.dit                  = model_pool.fetch_model("qwen_image_dit")
        pipe.vae                  = model_pool.fetch_model("qwen_image_vae")
        pipe.blockwise_controlnet = QwenImageBlockwiseMultiControlNet(
            model_pool.fetch_model("qwen_image_blockwise_controlnet", index="all")
        )

        if vlm_path is not None:
            from transformers import Qwen2Tokenizer, Qwen2VLProcessor
            pipe.tokenizer = Qwen2Tokenizer.from_pretrained(vlm_path)
            pipe.processor = Qwen2VLProcessor.from_pretrained(vlm_path)
            pipe.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                vlm_path,
                torch_dtype=torch_dtype,
                device_map="auto" if str(device) == "cuda" else None,
            ).eval()
            print(f"[Load] VLM + tokenizer + processor from: {vlm_path}")
        else:
            raise ValueError("vlm_path not defined")

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        inpaint_mask: Image.Image = None,
        inpaint_blur_size: int = None,
        inpaint_blur_sigma: float = None,
        height: int = 1328,
        width: int = 1328,
        seed: int = None,
        rand_device: str = "cpu",
        num_inference_steps: int = 20,
        exponential_shift_mu: float = None,
        blockwise_controlnet_inputs: list[ControlNetInput] = None,
        eligen_entity_prompts: list[str] = None,
        eligen_entity_masks: list[Image.Image] = None,
        eligen_enable_on_negative: bool = False,
        edit_image: Image.Image = None,
        edit_image_auto_resize: bool = True,
        edit_rope_interpolation: bool = False,
        zero_cond_t: bool = False,
        layer_input_image: Image.Image = None,
        layer_num: int = None,
        context_image: Image.Image = None,
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        progress_bar_cmd=tqdm,
        output_txt_path: str = None,
    ):
        self._last_decoder_outputs = None

        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            dynamic_shift_len=(height // 16) * (width // 16),
            exponential_shift_mu=exponential_shift_mu,
        )

        inputs_posi   = {"prompt": prompt}
        inputs_nega   = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "inpaint_mask": inpaint_mask, "inpaint_blur_size": inpaint_blur_size,
            "inpaint_blur_sigma": inpaint_blur_sigma,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "blockwise_controlnet_inputs": blockwise_controlnet_inputs,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "eligen_entity_prompts": eligen_entity_prompts,
            "eligen_entity_masks": eligen_entity_masks,
            "eligen_enable_on_negative": eligen_enable_on_negative,
            "edit_image": edit_image, "edit_image_auto_resize": edit_image_auto_resize,
            "edit_rope_interpolation": edit_rope_interpolation,
            "context_image": context_image,
            "zero_cond_t": zero_cond_t,
            "layer_input_image": layer_input_image,
            "layer_num": layer_num,
            "output_txt_path": output_txt_path,
        }

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id,
            )
            inputs_shared["latents"] = self.step(
                self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared
            )

        self.load_models_to_device(['vae'])
        image = self.vae.decode(
            inputs_shared["latents"], device=self.device,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        )
        if layer_num is None:
            image = self.vae_output_to_image(image)
        else:
            image = [self.vae_output_to_image(i, pattern="C H W") for i in image]
        self.load_models_to_device([])
        return image


# ─────────────────────────────────────────────
# QwenImageEdit
# ─────────────────────────────────────────────
class QwenImageEdit:
    def __init__(
        self,
        model_configs: list[ModelConfig],
        vlm_path: Optional[str] = None,
        vae_config: ModelConfig = None,
        dit_lora_path: Optional[str] = None,
        vlm_lora_path: Optional[str] = None,
        device: Union[str, torch.device] = get_device_type(),
        torch_dtype: torch.dtype = torch.bfloat16,
        vram_limit: float = None,
        seed: int = 42,
    ):
        self.vlm_path      = vlm_path
        self.vae_config    = vae_config
        self.dit_lora_path = dit_lora_path
        self.vlm_lora_path = vlm_lora_path
        self.device        = device
        self.torch_dtype   = torch_dtype
        self.seed          = seed

        self._set_seed()
        self.pipeline = self._init_model(model_configs, vram_limit)

    def _init_model(self, model_configs, vram_limit):
        pipeline = QwenImagePipeline.from_pretrained(
            torch_dtype=self.torch_dtype,
            device=self.device,
            model_configs=model_configs,
            vlm_path=self.vlm_path,
            vae_config=self.vae_config,
            vram_limit=vram_limit,
        )
        print("[Load] Base pipeline loaded")

        # ── DiT LoRA ──────────────────────────────────
        if self.dit_lora_path:
            target_modules = [
                "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
                "attn.to_add_out", "attn.to_k", "attn.to_out.0",
                "attn.to_q", "attn.to_v",
            ]
            lora_config = LoraConfig(
                r=32, lora_alpha=64, init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            pipeline.dit = get_peft_model(pipeline.dit, lora_config)
            pipeline.dit.load_adapter(self.dit_lora_path, adapter_name="default", is_trainable=False)
            pipeline.dit.set_adapter("default")
            print(f"[Load] DiT LoRA: {self.dit_lora_path}")

        # ── VLM LoRA ──────────────────────────────────
        if self.vlm_lora_path:
            pipeline.text_encoder = PeftModel.from_pretrained(
                pipeline.text_encoder, self.vlm_lora_path
            )
            print(f"[Load] VLM LoRA: {self.vlm_lora_path}")

        return pipeline

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def calc_hw(
        orig_w: int, orig_h: int,
        target_pixels: int = 1024 * 1024,
        multiple: int = 16,
    ) -> tuple[int, int]:
        import math
        ratio = orig_w / orig_h
        h = int(math.sqrt(target_pixels / ratio))
        w = int(h * ratio)
        h = max(multiple, round(h / multiple) * multiple)
        w = max(multiple, round(w / multiple) * multiple)
        return h, w

    def inference(
        self,
        image_path: str,
        output_txt_path: str = None,
        height: int = None,
        width: int = None,
        num_inference_steps: int = 20,
    ) -> Image.Image:
        self._set_seed()

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if height is None or width is None:
            height, width = self.calc_hw(image.width, image.height)
            print(f"[AutoResize] orig=({image.width}x{image.height}) -> h={height}, w={width}")

        image = [image.resize((width, height))]

        with torch.inference_mode():
            output = self.pipeline(
                prompt=user_prompt,
                negative_prompt="",
                cfg_scale=1.0,
                edit_image=image,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                output_txt_path=output_txt_path,
            )

        return output


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit_path",            type=str, required=True,
                        help="DiT safetensors path")
    parser.add_argument("--vlm_path",            type=str, required=True,
                        help="VLM checkpoint folder")
    parser.add_argument("--vae_path",            type=str, required=True,
                        help="VAE safetensors path")
    parser.add_argument("--dit_lora_path",       type=str, default=None)
    parser.add_argument("--vlm_lora_path",       type=str, default=None)
    parser.add_argument("--seed",                type=int, default=1)
    parser.add_argument("--image_path",          type=str, required=True)
    parser.add_argument("--output_folder",       type=str, required=True)
    parser.add_argument("--height",              type=int, default=None)
    parser.add_argument("--width",               type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    args = parser.parse_args()

    dit_config = ModelConfig(
        model_id="Qwen/Qwen-Image-Edit-2509",
        origin_file_pattern=args.dit_path,
    )
    vae_config = ModelConfig(
        model_id="Qwen/Qwen-Image",
        origin_file_pattern=args.vae_path,
    )

    qwen_image_edit = QwenImageEdit(
        model_configs=[dit_config],
        vlm_path=args.vlm_path,
        vae_config=vae_config,
        dit_lora_path=args.dit_lora_path,
        vlm_lora_path=args.vlm_lora_path,
        seed=args.seed,
    )

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "txt"), exist_ok=True)

    filename        = os.path.basename(args.image_path)
    stem            = os.path.splitext(filename)[0]
    output_img_path = os.path.join(args.output_folder, filename)
    output_txt_path = os.path.join(args.output_folder, "txt", stem + ".txt")

    output_image = qwen_image_edit.inference(
        image_path=args.image_path,
        output_txt_path=output_txt_path,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
    )
    output_image.save(output_img_path)
    print(f"[Saved] Image: {output_img_path}")

    decoder_outputs = getattr(qwen_image_edit.pipeline, "_last_decoder_outputs", None)
    if decoder_outputs is not None:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(decoder_outputs)
        print(f"[Saved] Text:  {output_txt_path}")
    else:
        print("[WARN] No decoder_outputs")


if __name__ == "__main__":
    main()


# python infer.py \
# --dit_path "models/SmartPhotoCrafter/dit.safetensors" \
# --vlm_path "models/SmartPhotoCrafter/vlm/" \
# --vae_path "models/Qwen-Image-Edit-2509/vae/diffusion_pytorch_model.safetensors" \
# --dit_lora_path "models/SmartPhotoCrafter/dit_lora" \
# --vlm_lora_path "models/SmartPhotoCrafter/vlm_lora" \
# --image_path "example/input/test.png" \
# --output_folder "example/output/"
