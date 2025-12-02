# python transfer_attack.py \
#     --model_name entrep \
#     --dataset_name entrep \
#     --transfer_dir "/d/thesis_result\resutls\ssl_entrepclip\ENTrep\dct_f=None_attack_nameES_1_Lambda_epsilon=0.1_norm=linf_mode=pre_transform_seed=22520691" \
#     --index_path evaluate_result/common_entrep.txt \
#     --prompt_path evaluate_result/model_name=entrep_ssl_dataset=entrep_prompt.json \
#     # --pretrained "/d/thesis_result\pretrained_finetune\entrep_base_multi_modal_ssl_finetuning.pt" \
#     --mode 'pre_transform' \

python transfer_attack.py \
    --model_name entrep \
    --dataset_name entrep \
    --transfer_dir "/d/thesis_result\resutls\scratch_entrepclip\ENTrep\dct_f=None_attack_nameES_1_Lambda_epsilon=0.08_norm=linf_mode=post_transform_seed=22520691" \
    --index_path evaluate_result/common_entrep.txt \
    --prompt_path evaluate_result/model_name=entrep_ssl_dataset=entrep_prompt.json \
    --pretrained "/d/thesis_result\pretrained_finetune\entrep_base_multi_modal_ssl_finetuning.pt" \
    --visual_backbone_mode 'ssl' \
    --mode 'post_transform' \


    