# 载入所需要的库
import pandas as pd
import os
from radiomics import featureextractor
import cv2
import SimpleITK as sitk
import numpy as np

def feature_extraction(image_dir, label_dir, para_path, csvs_path):
    
    # 初始化特征提取器
    extractor = featureextractor.RadiomicsFeatureExtractor(para_path, encodings='utf-8')

    # 创建变量以存放特征
    features_dict = dict()
    df = pd.DataFrame()

    # 遍历文件夹
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            
            # 在label_dir内查找是否有当前图像的掩膜
            label_path = os.path.join(label_dir, os.path.relpath(root, image_dir), file)
            if os.path.exists(label_path):
                print("{} is extracting".format(file))
                image_path = os.path.join(image_dir, os.path.relpath(root, image_dir), file)
                
                # 读取图像及掩膜
                img_arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                mask_arr = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                image = sitk.GetImageFromArray(img_arr)
                mask = sitk.GetImageFromArray(mask_arr)

                # 二值化
                mask = sitk.BinaryThreshold(mask, 1, 255, 1, 0)

                # 尝试特征提取，若提取失败则报错后继续执行下一文件
                try:
                    features = extractor.execute(image, mask)
                    features_dict['index'] = os.path.relpath(root, image_dir)
                    for key, value in features.items():
                        features_dict[key] = value
                    df = pd.concat([df, pd.DataFrame([features_dict])], ignore_index=True)
                except Exception as e:
                    print(f"Error extracting features from {file}: {e}")
                    continue

    # 将特征保存至csv文件                
    if not df.empty:
        df.columns = features_dict.keys()
        df.to_csv(csv_path, index=False)
        print('Done')
    else:
        print('No features were extracted.')

# 输入及输出路径（需要进行调整）
para_path = 'C:/Users/xiaoping/Desktop/apple/code/Params_photo.yaml'
image_dir = r"C:\Users\xiaoping\Desktop\fushi\data\photo_rename"
mask_dir = r"C:\Users\xiaoping\Desktop\fushi\data\mask_rename"
csv_path = r"C:\Users\xiaoping\Desktop\fushi\data\result\radiomics_features_Photo_fushi.csv"
feature_extraction(image_dir, mask_dir, para_path, csv_path)