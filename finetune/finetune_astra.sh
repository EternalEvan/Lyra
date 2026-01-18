export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

# Store videos generated while finetuning
# mkdir videos

# sudo apt-get update
# yes | sudo apt-get install python3-tk

# Download reward model
# git clone https://github.com/tgxs002/HPSv2.git
# cd HPSv2
# pip install -e . 
# cd ..

CUDA_VISIBLE_DEVICES=0 NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=1 --master_port 19003 \
    finetune/train_grpo_astra.py \
    --seed 42 \
    --pretrained_model_name_or_path /home/zhuyixuan05/ReCamMaster/models/Wan-AI/Wan2.1-T2V-1.3B \
    --vae_model_path /home/zhuyixuan05/ReCamMaster/models/Wan-AI/Wan2.1-T2V-1.3B \
    --cache_dir /share_zhuyixuan05/zhuyixuan05/dancegrpo/data/.cache \
    --data_json_path /share_zhuyixuan05/zhuyixuan05/dancegrpo/data/rl_embeddings/videos2caption.json \
    --train_batch_size 2 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 24 \
    --max_train_steps 1000 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 5 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir /mnt/data/louis_crq/DanceGRPO/data/outputs/grpo \
    --h 480 \
    --w 832 \
    --t 33 \
    --sampling_steps 20 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --gradient_checkpointing \
    --num_generations 12 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --cfg_infer 5.0 \
    --our_checkpoint_path /mnt/data/louis_crq/astra2/playground/checkpoints/step13336_origin_other_continue3.ckpt \
    --moe_num_experts 3 \
    --moe_top_k 1 \
    --moe_hidden_dim 128 \
    --dataset_path /mnt/data/louis_crq/preprocessed_data/SpatialVID_Wan2/manifest.json \
    --lora_rank 32 \
    --lora_alpha 128 \