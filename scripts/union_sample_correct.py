import json
import random

# =======================
# 1. Đọc dữ liệu
# =======================
path_1 = r"D:/thesis_result/Vision-language-Models-in-Medical-Image-Analysis/evaluate_result/model_name=entrep_dataset=entrep_n_prompt=3.json"
path_2 = r"D:/thesis_result/Vision-language-Models-in-Medical-Image-Analysis/evaluate_result/model_name=entrep_ssl_dataset=entrep_n_prompt=5.json"

data_1 = json.load(open(path_1, "r"))
data_2 = json.load(open(path_2, "r"))

# =======================
# 2. Lấy index dự đoán đúng
# =======================
idxs_1 = [item['index'] for item in data_1 if item['class_pred_id'] == item['gt_pred_id']]
idxs_2 = [item['index'] for item in data_2 if item['class_pred_id'] == item['gt_pred_id']]
print(len(idxs_1), len(idxs_2))
# =======================
# 3. Lấy giao
# =======================
giao = list(set(idxs_1) & set(idxs_2))

print("len giao:", len(giao))

# =======================
# 4. Gom nhóm theo class trong tập giao
# =======================
by_class = {'0': [], '1': [], '2': [], '3': []}

for item in data_1:  # dùng data_1 hoặc data_2 đều được
    idx = item['index']
    if idx in giao:
        cls = str(item['gt_pred_id'])
        by_class[cls].append(idx)

# =======================
# 5. In số lượng mỗi class
# =======================
for cls, lst in by_class.items():
    print(f"Class {cls} trong giao có {len(lst)} mẫu")

# =======================
# 6. Lấy tối đa có thể cho mỗi class (max per class)
# =======================
selected = []
MAX_PER_CLASS = 500  # giới hạn upper bound

for cls, lst in by_class.items():
    if len(lst) > MAX_PER_CLASS:
        chosen = random.sample(lst, MAX_PER_CLASS)
    else:
        chosen = lst[:]  # lấy hết nếu không đủ
    selected.extend(chosen)
    print(f"→ Class {cls}: lấy {len(chosen)} mẫu")

print("Tổng số index được chọn:", len(selected))

# =======================
# 7. Lưu ra file
# =======================
with open("selected_indices_from_giao.txt", "w") as f:
    for idx in selected:
        f.write(f"{idx}\n")

print("Đã lưu selected_indices_from_giao.txt")
