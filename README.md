# A2G: Leveraging Intuitive Physics for Force-Efficient Robotic Grasping

This is the code of "A2G: Intuitive Physics Helps Robot Grasp Better to Improve the Manipulation Performance"

We will update the way to use it as soon as possible.

[paper](https://ieeexplore.ieee.org/abstract/document/10531637), [video](https://www.youtube.com/watch?v=-j9SkiYL1Rc&t=4s)

## Citation
If you find our work useful, please cite it.
```
@ARTICLE{10531637,
  author={Cheng, Yaofeng and Liu, Shengkai and Zha, Fusheng and Guo, Wei and Du, Haoyu and Wang, Pengfei and Bing, Zhenshan},
  journal={IEEE Robotics and Automation Letters}, 
  title={A2G: Leveraging Intuitive Physics for Force-Efficient Robotic Grasping}, 
  year={2024},
  volume={9},
  number={7},
  pages={6376-6383},
  keywords={Task analysis;Force;Physics;Point cloud compression;Market research;Grasping;Grippers;Deep learning;grasp position logic;task-oriented grasp},
  doi={10.1109/LRA.2024.3401675}}
```


## Train and inference
```bash
cd A2G

# train
bash ./scripts/run_handle.sh

# inference
bash ./scripts/inference_handle.sh
```

new dataset can be generated with `run_handle.sh`, if you don't want to generate new data, just add the `dataset_path` in the sh file.
