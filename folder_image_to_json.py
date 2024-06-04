import os
import json

def create_dataset_json(data_dir):
    dataset = {'labels': [], 'data': []}

    # Lấy danh sách các nhãn từ tên thư mục con
    labels = sorted(os.listdir(data_dir))

    for label_id, label in enumerate(labels):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            image_files = os.listdir(label_path)
            for image_file in image_files:
                # Tạo đường dẫn đầy đủ đến tệp hình ảnh
                image_path = os.path.join(label_path, image_file)
                # Thêm dữ liệu vào tập dữ liệu
                dataset['data'].append({'filename': image_path, 'label': label_id})
        # Thêm nhãn vào danh sách nhãn
        dataset['labels'].append(label)

    return dataset

def save_json(dataset, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(dataset, json_file)

# Thay đổi đường dẫn của thư mục dữ liệu của bạn
data_directory = './data/vietnameseimage'

# Tạo dataset JSON
dataset = create_dataset_json(data_directory)

# Lưu dataset JSON vào một tệp
save_json(dataset, 'dataset.json')
