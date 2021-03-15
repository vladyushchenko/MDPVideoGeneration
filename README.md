# MDP for Video Generation
This repository contains code to reproduce results from paper **Markov Decision Process for Video Generation**

## Prerequisites
The project uses Python 3.6, PyTorch 0.4.1 and Torchvision 0.2.1

TensorFlow is optional, used only to calculate FVD scores.

## Installation
Create a conda environment using:

`conda env create -f environment.yml`

Note: there may be some additional dependencies that are no longer needed

Optionally, you can install the package with the following command

`python setup.py install`

## UCF-101 data extraction
1) download UCF-101 dataset from [this website](`https://www.crcv.ucf.edu/data/UCF101.php`)
2) unzip the archive to the target folder, which will contain raw UCF-101 videos.
3) generate train/test splits from `ucfTrainTestlist`. We use `*01.txt` splits per default..
   Use `mdp_video/transform/ucf_move_files.py` script for that.
4) transform all video to separate images (you need to have FFmpeg installed).
   Use `mdp_video/transform/ucf_extract_files.py` script for that.

Launch templates for both scripts can be found in `.run` folder

## Training
For training, we provide `mdp_video/launcher_template.sh` template training script.
As a rule, you just need to set `dataset_root` location to start training,
but you can also try different configurations of training parameters.

Some parameters are pre-set to train specific MDP model with `sigma=0.7` and `beta=0.9`,
for more info, please refer to the paper or to `train.py` help to set your specific experiment.

## Inference (generating videos and image)
```python
python -m mdp_video.generate_videos
--model <path to generator checkpoint *.pytorch>
--output_format gif
--n_frames 64
--num_videos 50
--col_videos 10
--save_images
--fix_seed
--save_mosaic
--save_diff
--save_chunks
--add_counter
--img_ext png
```

## Reproducing metrics
To reproduce IS metrics, please refer to the `https://github.com/pfnet-research/tgan` repository.
We include the normalization file, needed for the IS calculations.

To reproduce FVD and temporal metrics, use PyCharm configurations from `.run` folder or these commands directly

Temporal metrics
```python
python -m mdp_video.metrics.temporal_metric
--model <path_to_model>
--location <optional_path_to_dataset>
--mode generator
--num_workers 0
--metric <psnr|dssim|ssim>
--calc_iter 256
--video_length 64
```

FVD from `https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py` repo.
```python
python -m mdp_video.metrics.frechet_video_distance.py
--model <path_to_model>
--mode generator
--dataset_loc <path_to_dataset>
--calc_iter 256
--seed 0
--num_workers 0
--video_length 64
--cuda
```

## Download links
Final model checkpoints are available [here](https://drive.google.com/drive/folders/1O7WzXIApMliJ00iSlthz3UEsiI-ZHfHS?usp=sharing)

## License
The code is provided for research purposes only. (Apache 2.0)

## Citation
If you find this paper useful in your research, please consider citing the paper

```
@article{Yushchenko2019MarkovDP,
  title={Markov Decision Process for Video Generation},
  author={Vladyslav Yushchenko and Nikita Araslanov and Stefan Roth},
  journal={2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)},
  year={2019},
  pages={1523-1532}
}
```
