from absl import app
import os
from pathlib import Path
import re

# mkdir UAV; cd UAV
# ~/.local/bin/gdown https://drive.google.com/uc?id=1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc
# unzip UAV-benchmark-M.zip
# ~/.local/bin/gdown https://drive.google.com/uc?id=19498uJd7T9w4quwnQEy62nibt3uyT9pq
# unzip UAV-benchmark-MOTD_v1.0.zip 

def main(argv):
    data_dir = Path('UAV-benchmark-M').absolute()
    labels_dir = Path('UAV-benchmark-MOTD_v1.0/GT').absolute()
    
    for f in os.listdir(data_dir):
        curr_label1 = labels_dir / (f + "_gt_whole.txt")
        curr_label2 = labels_dir / (f + "_gt_ignore.txt")
        curr_label3 = labels_dir / (f + "_gt.txt")
        new_label1 = data_dir / f / "gt_whole.txt"
        new_label2 = data_dir / f / "gt_ignore.txt"
        new_label3 = data_dir / f / "gt.txt"
        if curr_label1.exists():
            Path(curr_label1).rename(new_label1)
        if curr_label2.exists():
            Path(curr_label2).rename(new_label2)
        if curr_label3.exists():
            Path(curr_label3).rename(new_label3) 
                
        
if __name__ == "__main__":
    app.run(main)