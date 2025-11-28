CUDA_VISIBLE_DEVICES=3 python main_atttack.py \
    --model_name medclip \
    --dataset_name rsna \
    --mode pre_transform \
    --index_path evaluate_result/medclip_biomedclip_same_correct.txt \
    --prompt_path evaluate_result/ssl_model_name=medclip_dataset=rsna_prompt.json \
    --attacker_name PGD \
    --epsilon 0.03 \
    --norm linf \
    --PGD_steps 100 \
    # --visual_backbone_pretrained "checkpoints/target_model/medclip_ssl_backbone.pth" \
    # --pretrained
    # --visual_backbone_mode 'ssl'