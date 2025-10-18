import json
import random

path = "/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/evaluate_result/model_name=medclip_dataset=covid_n_prompt=5.json"
# Đọc file JSON
with open(path, "r") as f:
    data = json.load(f)
print(len(data))
# Lọc các mẫu có dự đoán đúng
correct = [item for item in data if item["class_pred_id"] == item["gt_pred_id"]]
print(len(correct) / len(data))
# Gom nhóm theo class
by_class = {
    '0': [],
    '1': []
}
for item in correct:
    by_class[str(item['gt_pred_id'])].append(item['index'])
    

# Lấy 500 mẫu ngẫu nhiên mỗi lớp (nếu đủ)
selected = []
for c, indices in by_class.items():
    chosen = random.sample(indices, min(500, len(indices)))
    selected.extend(chosen)

# Đảm bảo chỉ có 1000 dòng (nếu có nhiều hơn)
selected = selected[:1000]

# Ghi ra file
with open("selected_indices.txt", "w") as f:
    for idx in selected:
        f.write(f"{idx}\n")

print(f"Saved {len(selected)} indices to selected_indices.txt")
