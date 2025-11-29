CUDA_VISIBLE_DEVICES=3 python transfer_attack.py \
    --model_name medclip \
    --dataset_name rsna \
    --transfer_dir attack_new/medclip/rsna/ssl_dct_f=None_attack_namePGD_epsilon=0.05_norm=linf_mode=post_transform_seed=42 \
    --index_path evaluate_result/medclip_biomedclip_same_correct.txt \
    --prompt_path evaluate_result/ssl_model_name=medclip_dataset=rsna_prompt.json \

# CUDA_VISIBLE_DEVICES=3 python transfer_attack.py \
#     --model_name medclip \
#     --dataset_name rsna \
#     --transfer_dir attack_new/medclip/rsna/scratch_dct_f=None_attack_namePGD_epsilon=0.1_norm=linf_mode=post_transform_seed=42 \
#     --index_path evaluate_result/medclip_biomedclip_same_correct.txt \
#     --prompt_path evaluate_result/ssl_model_name=medclip_dataset=rsna_prompt.json \
#     --pretrained "checkpoints/target_model/medclip_ssl_multi_modal_finetuning.pt" \
#     --visual_backbone_mode 'ssl'

    