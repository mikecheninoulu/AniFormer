# AniFormer

This is the PyTorch implementation of our BMVC 2021 paper [AniFormer: Data-driven 3D Animation with Transformer](https://arxiv.org/abs/2110.10533).
[Haoyu Chen](https://scholar.google.com/citations?user=QgbraMIAAAAJ&hl=en), [Hao Tang](https://scholar.google.com/citations?user=9zJkeEMAAAAJ&hl=en), [Nicu Sebe](https://scholar.google.it/citations?user=tNtjSewAAAAJ&hl=en), [Guoying Zhao](https://scholar.google.com/citations?user=hzywrFMAAAAJ&hl=en). <br>

<img src="Capture.PNG" width="400" height="300">

#### Citation

If you use our code or paper, please consider citing:
```
@inproceedings{chen2021AniFormer,
  title={AniFormer: Data-driven 3D Animation withTransformer},
  author={Chen, Haoyu and Tang, Hao and Sebe, Nicu and Zhao, Guoying},
  booktitle={BMVC},
  year={2021}
}
```

## Dependencies

Requirements:
- python3.6
- numpy
- pytorch==1.1.0 and above
- [trimesh](https://github.com/mikedh/trimesh)

## Dataset preparation
Please download DFAUST dataset from [DFAUST link](https://dfaust.is.tue.mpg.de/) for training the model.

Generate the driving sequence based on our script:

TBD

## Training
The usage of our code is easy, just run the code below.
```
python train.py
```

## Acknowledgement
Part of our code is based on 

3D transfer: [NPT](https://github.com/jiashunwang/Neural-Pose-Transfer)，

Transformer framework: (https://github.com/lucidrains/vit-pytorch) 

Many thanks!

## License
MIT-2.0 License
