CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_path "ckpt/Qwen-Image-Edit-2509" \
    --dit_path "ckpt/DiT.safetensors" \
    --vlm_path "ckpt/text_encoder" \
    --prompt "slightly decrease contrast;moderately decrease light&exposure" \
    --image_path "example/841012.png" \
    --output_folder "example/output/manual" \
    --seed 42 \