from diffusers import QwenImageEditPipeline
from typing import Union, List, Optional, Tuple
import torch
import torch.nn.functional as F
import os
from PIL import Image
from diffsynth import load_state_dict
from peft import LoraConfig, get_peft_model
from peft import PeftModel
import random
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration
import argparse

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

# 4. final input
user_prompt = (
    f"{system_prompt}\n"
    f"{Quality}\n"
    f"{Color}\n"
    f"{Composition}\n"
    f"{Aesthetic}\n"
    f"{Evaluation}\n"
    f"{Suggestions}\n"
)


class CustomQwenImageEditPipeline(QwenImageEditPipeline):

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]

        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)
        prompt_ids, prompt_mask = model_inputs["input_ids"], model_inputs["attention_mask"]

        if "You are an expert in image quality and aesthetic evaluation." in prompt[-1]:
            print("Auto editing mode")
            with torch.no_grad():
                outputs = self.text_encoder.generate(
                    **model_inputs, 
                    max_new_tokens=1280, 
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                )
            # hidden_states
            all_hidden_states = outputs.hidden_states
            completion_hidden = torch.cat([hs[-1] for hs in all_hidden_states[1:]], dim=1)
            prompt_hidden = all_hidden_states[0][-1]
            hidden_states = torch.cat([prompt_hidden, completion_hidden], dim=1)

            # Decoded text output
            prompt_completion_ids = outputs.sequences
            completion_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(prompt_ids, prompt_completion_ids)]
            decoder_outputs = self.processor.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print("decoder_outputs", decoder_outputs)

            # Build full attention mask
            prompt_mask = model_inputs.attention_mask
            completion_hidden_len = completion_hidden.shape[1]
            completion_mask = torch.ones(
                prompt_mask.shape[0], completion_hidden_len,
                dtype=prompt_mask.dtype, device=prompt_mask.device
            )
            full_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

            # Extract hidden states
            split_hidden_states = self._extract_masked_hidden(hidden_states, full_attention_mask)
            split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
            )
            encoder_attention_mask = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
            )

            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        else:
            print("Manual editing mode: ", prompt)
            # Use encoder output
            outputs = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states[-1]
            split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
            split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
            )
            encoder_attention_mask = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
            )

            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask


class QwenImageEdit():
    def __init__(
        self,
        model_path: str,
        dit_path: Optional[str] = None,
        vlm_path: Optional[str] = None,
        dit_lora_path: Optional[str] = None,
        vlm_lora_path: Optional[str] = None,
        seed: int = 42,
    ):
        self.model_path = model_path
        self.vlm_path = vlm_path
        self.dit_path = dit_path
        self.dit_lora_path = dit_lora_path
        self.vlm_lora_path = vlm_lora_path
        self.seed = seed

        self._set_seed()
        self.pipeline = self._init_model()


    def _init_model(self):
        pipeline = CustomQwenImageEditPipeline.from_pretrained(self.model_path)
        print(f"[Load] Base model: {self.model_path}")
        if self.dit_path:
            state_dict = load_state_dict(self.dit_path)
            pipeline.transformer.load_state_dict(state_dict)
            print(f"[Load] DiT weights: {self.dit_path}")
        if self.dit_lora_path:
            target_modules = [
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "attn.to_k",
                "attn.to_out.0",
                "attn.to_q",
                "attn.to_v",
            ]
            transformer_lora_config = LoraConfig(
                r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules
            )
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
            pipeline.transformer.load_adapter(self.dit_lora_path, adapter_name="default", is_trainable=False)
            print(f"[Load] DiT LoRA: {self.dit_lora_path}")
        if self.vlm_path:
            mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.vlm_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0"
            )
            pipeline.text_encoder = mllm
            print(f"[Load] VLM weights: {self.vlm_path}")
        if self.vlm_lora_path:
            mllm = PeftModel.from_pretrained(pipeline.text_encoder, self.vlm_lora_path)
            pipeline.text_encoder = mllm
            print(f"[Load] VLM LoRA: {self.vlm_lora_path}")

        pipeline = pipeline.to(torch.bfloat16)
        pipeline = pipeline.to("cuda")

        return pipeline


    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def inference(self, image_path: str, prompt: str = None):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if prompt is None:
            prompt = user_prompt

        image = Image.open(image_path).convert("RGB")
        inputs = {
            "image": image,
            "prompt": prompt,
            "true_cfg_scale": 1.0,
            "negative_prompt": "",
            "num_inference_steps": 20,
        }

        with torch.inference_mode():
            output = self.pipeline(**inputs)
            output_image = output.images[0]

        return output_image
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dit_path", type=str, default=None)
    parser.add_argument("--vlm_path", type=str, default=None)
    parser.add_argument("--dit_lora_path", type=str, default=None)
    parser.add_argument("--vlm_lora_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Inference arguments
    parser.add_argument("--image_path", type=str, required=True, help="Input image path")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder path")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Editing instruction (if not provided, auto editing mode is used)")

    args = parser.parse_args()

    # 1. Load model
    qwen_image_edit = QwenImageEdit(
        model_path=args.model_path,
        dit_path=args.dit_path,
        vlm_path=args.vlm_path,
        dit_lora_path=args.dit_lora_path,
        vlm_lora_path=args.vlm_lora_path,
        seed=args.seed,
    )

    # 2. Run inference
    os.makedirs(args.output_folder, exist_ok=True)
    filename = os.path.basename(args.image_path)
    output_path = os.path.join(args.output_folder, filename)
    output_image = qwen_image_edit.inference(args.image_path, args.prompt)
    output_image.save(output_path)
    print("Image saved at", output_path)


if __name__=="__main__":
    main()
