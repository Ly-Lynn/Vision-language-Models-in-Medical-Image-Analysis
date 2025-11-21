path_1 = "/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/evaluate_result/selected_indices_rsna_biomedclip.txt"
path_2 = "/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/evaluate_result/selected_indices_rsna_ssl_biomedclip.txt"
output_path = "common_biomedclip.txt"

def load_int_set(path):
    with open(path, "r") as f:
        return set(int(line.strip()) for line in f if line.strip())

# Load data
set1 = load_int_set(path_1)
set2 = load_int_set(path_2)

# Intersection
common = sorted(list(set1 & set2))

# Write to file
with open(output_path, "w") as f:
    for idx in common:
        f.write(str(idx) + "\n")

print(f"Done. Common count: {len(common)}")
print(f"Saved to: {output_path}")
