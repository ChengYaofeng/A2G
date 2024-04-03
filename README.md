# A2G: Intuitive Physics Helps Robot Grasp Better to Improve the Manipulation Performance

This is the code of "A2G: Intuitive Physics Helps Robot Grasp Better to Improve the Manipulation Performance"

We will update the way to use it as soon as possible.

[paper]()

## Citation
If you find our work useful, please cite it.


## Train and inference
```bash
cd A2G

# train
bash ./scripts/run_handle.sh

# inference
bash ./scripts/inference_handle.sh
```

new dataset can be generated with `run_handle.sh`, if you don't want to generate new data, just add the `dataset_path` in the sh file.
