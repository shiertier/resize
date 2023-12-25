import cv2
import numpy as np
import os

def generate_csv_data():
    pairs = []
    #header = ['w', 'h', 'w*h', 'w/h', 'h/w'];1030000, 1067152

    for i in range(320, 4033, 32):
        for j in range(i, 4033, 32):
            product = i * j
            if 1030000 <= product <= 1067152:
                pairs.append([i, j, product, i / j, j / i])

    pairs = [[float(e) for e in row] for row in pairs]

    return pairs

def resize_crop_image(image_path, bucket_data):
    print(image_path)
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert not isinstance(img,type(None)), 'image not found'
    height, width = img.shape[:2]

    aspect_ratio = width / height

    if aspect_ratio < 1:
        idx = min(bucket_data, key=lambda x: abs(float(x[3]) - aspect_ratio))
        w, h = idx[0], idx[1]
    else:
        idx = min(bucket_data, key=lambda x: abs(x[4] - aspect_ratio))
        w, h = idx[1], idx[0]

    scale_ratio = max(w / width, h / height)
    resized_img = cv2.resize(img, None, fx=scale_ratio, fy=scale_ratio)

    crop_x = resized_img.shape[1] - w
    crop_y = resized_img.shape[0] - h
    print(crop_x,crop_y)

    if crop_x > 0:
        if crop_x % 2 == 0:
            left_crop = crop_x // 2
            right_crop = left_crop
        else:
            left_crop = crop_x // 2
            right_crop = left_crop + 1
    else:
        left_crop = 0
        right_crop = 0

    if crop_y > 0:
        if crop_y % 2 == 0:
            top_crop = crop_y // 2
            bottom_crop = top_crop
        else:
            top_crop = crop_y // 2
            bottom_crop = top_crop + 1
    else:
        top_crop = 0
        bottom_crop = 0

    print(left_crop,right_crop,top_crop,bottom_crop)
    crop_yy = resized_img.shape[0]-bottom_crop
    crop_xx = resized_img.shape[1]-right_crop
    print(crop_yy,crop_xx)
    cropped_img = resized_img[int(top_crop):int(crop_yy), int(left_crop):int(crop_xx)]
    return cropped_img

def resize_crop_image_batch(input_folder, output_folder, bucket_data, num=1024):
    rm_folder = os.path.join(output_folder, "rm")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(rm_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            image_path = os.path.join(root, file_name)
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            height, width, _ = img.shape
            
            total_pixels = height * width
            if total_pixels < num ** 2:
                new_file_path = os.path.join(rm_folder, file_name)
                os.rename(image_path, new_file_path)
            else:
                new_file_path = os.path.join(output_folder, file_name)
                if os.path.exists(new_file_path):
                    base_file_name, extension = os.path.splitext(file_name)
                    i = 1
                    while os.path.exists(os.path.join(output_folder, f"{base_file_name}_{i}{extension}")):
                        i += 1
                    new_file_path = os.path.join(output_folder, f"{base_file_name}_{i}{extension}")
                
                img = resize_crop_image(image_path, bucket_data)
                cv2.imwrite(new_file_path, img)

input_folder = r"E:\a1\Aak"
output_folder = r"E:\a2\Aak"
bucket_data = generate_csv_data()
resize_crop_image_batch(input_folder, output_folder, bucket_data, 1024)