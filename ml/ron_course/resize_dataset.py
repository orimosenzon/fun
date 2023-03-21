import os 

from PIL import Image

dir1 = 'dataset_orig'
dir2 = 'dataset'

for i, fname in enumerate(os.listdir(dir1)):
    img = Image.open(f'{dir1}/{fname}')
    img1 = img.resize((200, 200))
    img1.save(f'{dir2}/{fname}')
    if i % 10 == 0:
        print(i)
