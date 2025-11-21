CUDA_VISIBLE_DEVICES=1 python3 main_atttack.py --dataset_name entrep --model_name entrep \
    --index_path evaluate_result/selected_indices_entrep_entrep.txt \
    --prompt_path evaluate_result/model_name=entrep_ssl_dataset=entrep_prompt.json \
    --attacker_name ES_1_Lambda \
    --epsilon 0.1 \
    --norm linf \
    --max_evaluation 10000 \
    --mode pre_transform \
    --seed 22520692 \
    --out_dir attack_tanh_transform \
    --start_idx 0 \
    # --end_idx 200

CUDA_VISIBLE_DEVICES=1 python3 main_atttack.py --dataset_name entrep --model_name entrep \
    --index_path evaluate_result/selected_indices_entrep_entrep.txt \
    --prompt_path evaluate_result/model_name=entrep_ssl_dataset=entrep_prompt.json \
    --attacker_name ES_1_Lambda \
    --epsilon 0.08 \
    --norm linf \
    --max_evaluation 10000 \
    --mode pre_transform \
    --seed 22520692 \
    --out_dir attack_tanh_transform \
    --start_idx 0 \
    # --end_idx 200

CUDA_VISIBLE_DEVICES=1 python3 main_atttack.py --dataset_name entrep --model_name entrep \
    --index_path evaluate_result/selected_indices_entrep_entrep.txt \
    --prompt_path evaluate_result/model_name=entrep_ssl_dataset=entrep_prompt.json \
    --attacker_name ES_1_Lambda \
    --epsilon 0.05 \
    --norm linf \
    --max_evaluation 10000 \
    --mode pre_transform \
    --seed 22520692 \
    --out_dir attack_tanh_transform \
    --start_idx 0 \
    # --end_idx 200

CUDA_VISIBLE_DEVICES=1 python3 main_atttack.py --dataset_name entrep --model_name entrep \
    --index_path evaluate_result/selected_indices_entrep_entrep.txt \
    --prompt_path evaluate_result/model_name=entrep_ssl_dataset=entrep_prompt.json \
    --attacker_name ES_1_Lambda \
    --epsilon 0.03 \
    --norm linf \
    --max_evaluation 10000 \
    --mode pre_transform \
    --seed 22520692 \
    --out_dir attack_tanh_transform \
    --start_idx 0 \
    # --end_idx 200