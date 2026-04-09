CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_path "ckpt/Qwen-Image-Edit-2509" \
    --dit_path "ckpt/DiT.safetensors" \
    --vlm_path "ckpt/text_encoder" \
    --image_path "example/841012.png" \
    --output_folder "example/output/automatic" \
    --seed 42 \