# 🤖 YOLOv1 Paper Implementation 
This is repository contains the implementation and training scripts on GPU for the YOLOv1 object detection model according to the details in the paper using the PyTorch framework.

# 🧰 Dependencies
```
- python=3.10.13
- matplotlib=3.8.0
- numpy=1.26.3
- pandas=2.1.4
- pytorch=2.2.0
- pytorch-cuda=12.1
- tensorboard=2.10.0
- torchaudio==2.2.0
- torchvision==0.17.0
```

# 🛠️ Demos
![image](https://github.com/MingLongSu/YOLOv1-Implementation/assets/88013020/e6a47cf3-ff47-4d6f-88b5-7df36c166fd8)
![image](https://github.com/MingLongSu/YOLOv1-Implementation/assets/88013020/a2352d2d-d305-49c3-87da-7cfb89f31be2)

# 🏎️ Training
To run the training script, (use paths or) go to directory with ```run_training.py```. Then run script with the following:
```
python run_training.py 
--batch_size=4 
--epochs=135 
--save_epoch_freq=10 
--start_epoch=0 
--metaset_path=path/to/train/meta_data
--imgs_path=path/to/images_dir
--lbls_path=path/to/labels_dir
--output_dir=path/to/checkpoints_dir
--log_dir=path/to/tensorboards_output_dir
```

# 👷 Testing 
To run the testing scripts, (use paths or) go to tests directory. Then run any of the training scripts with the following:
```
python my_test_file.py 
```

# ❤️ Special Thanks
Much love to [Aladdin Persson](https://github.com/aladdinpersson) for his YouTube series on computer vision principles such as non-maximum suppression and explaining the YOLOv1 paper itself which helped with this implementation!
