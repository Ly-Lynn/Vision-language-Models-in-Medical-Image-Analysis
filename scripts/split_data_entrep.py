import os
import zipfile

def zip_folder(folder_path, output_zip):
    # Tạo file zip nén thư mục
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Duyệt toàn bộ file trong folder
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # relative path để giữ cấu trúc thư mục
                rel_path = os.path.relpath(file_path, folder_path)
                
                zipf.write(file_path, rel_path)

    print(f"Đã zip xong: {output_zip}")

# Cách dùng
folder = "/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/local_data/entrep"
output = "entrep_test.zip"

zip_folder(folder, output)