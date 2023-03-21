import os
import glob

cat_files = glob.glob(f'dataset/cat*')
dog_files = glob.glob(f'dataset/dog*')

for file in cat_files[:100]:
    os.system(f'cp {file} ds/{file[8:]}') 

for file in dog_files[:100]:
    os.system(f'cp {file} ds/{file[8:]}') 
