import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = ['https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Ariel_Sharon_Headshot.jpg/440px-Ariel_Sharon_Headshot.jpg']  # batch of images
# imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie


# import os
# import pandas as pd


# def read_annotations(path_to_label_txt):
#     """
#     Read the annotation files (txt) and create a data frame and a list that includes the frames' names.

#     Parameters
#     ----------
#     path_to_label_txt : str
#         The files location of the annotations.

#     Returns
#     -------
#     dataframe
#         A dataframe of the annotations includes 4 points of the bounding box of the annotation, category_id, and Frame name.
#     list
#         A list that includes the frames' names.
#     """
#     list_annotations_files = []
#     list_frame_names = []
#     columns_names = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'category_id']

#     # iterate over files in the directory
#     label_txt_files = os.listdir(path_to_label_txt)
#     for filename in label_txt_files:
#         df = pd.read_csv(os.path.join(path_to_label_txt, filename), sep=" ", header=None, names=columns_names)
#         frame_name = filename.split(".")[0]
#         df["Frame"] = frame_name
#         list_annotations_files.append(df)
#         list_frame_names.append(frame_name)

#     ann_train = pd.concat(list_annotations_files)
#     ann_train.reset_index(drop=True, inplace=True)

#     return ann_train, list_frame_names
