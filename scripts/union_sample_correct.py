import json
import random

# =======================
# 1. Đọc dữ liệu
# =======================
path_1 = "/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/evaluate_result/model_name=biomedclip_dataset=rsna_n_prompt=5.json"
path_2 = "/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/evaluate_result/model_name=medclip_dataset=rsna_n_prompt=5.json"

data_1 = json.load(open(path_1, "r"))
data_2 = json.load(open(path_2, "r"))

# =======================
# 2. Lấy index dự đoán đúng
# =======================
idxs_1 = [item['index'] for item in data_1 if item['class_pred_id'] == item['gt_pred_id']]
idxs_2 = [item['index'] for item in data_2 if item['class_pred_id'] == item['gt_pred_id']]

# =======================
# 3. Lấy giao
# =======================
giao = list(set(idxs_1) & set(idxs_2))

print("len giao:", len(giao))

# =======================
# 4. Gom nhóm theo class trong tập giao
# =======================
by_class = {
    '0': [],
    '1': [],
}

for item in data_1:  # dùng data_1 hoặc data_2 đều được
    idx = item['index']
    if idx in giao:
        cls = str(item['gt_pred_id'])
        if cls in by_class:
            by_class[cls].append(idx)

# =======================
# 5. Kiểm tra số lượng mỗi lớp trong giao
# =======================
for cls, lst in by_class.items():
    print(f"Class {cls} trong giao có {len(lst)} mẫu")
    if len(lst) < 500:
        raise ValueError(f"Lỗi: class {cls} chỉ có {len(lst)} mẫu trong giao — không đủ 500!")

# =======================
# 6. Chọn đúng 500/class
# =======================
selected = []
for cls, lst in by_class.items():
    chosen = random.sample(lst, 500)
    selected.extend(chosen)

print("Tổng số index được chọn:", len(selected))  # phải = 1000

# =======================
# 7. Lưu ra file
# =======================
with open("selected_indices_from_giao.txt", "w") as f:
    for idx in selected:
        f.write(f"{idx}\n")

print("Đã lưu selected_indices_from_giao.txt")
