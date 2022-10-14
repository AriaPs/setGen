# Data Set Construction for Paper

<p align="center">
<img src="readme_files/example.png" alt="Front readme image" width=400>
</p>

## SetUp

As we use [BlenderProc](https://github.com/DLR-RM/BlenderProc) for data set generation, run:

```bash
pip install blenderproc tqdm
``` 

You can test your BlenderProc pip installation by running:

```bash
blenderproc quickstart
```

We used [AmbientCG](https://ambientcg.com/) for wall and floor texture. Please download the textures from cc_textures provided script by blenderproc [here](https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/scripts/download_cc_textures.py).

Make sure that you have downloaded the `cctextures` before executing and moved it to `/resources/cctextures`.


## Usage

1. Edit `gen_train.bash`. If you use the same ordner structer as default param then you need to change only `number_of_scene`.

```bash
python ./BlenderProc/rerun.py number_of_scene start_index main.py  Path/To/resources/haven_furnitures/ Path/To/resources/cctextures/ Path/To/resources/haven_objects/ resources/material/ ./output_train
``` 


2. Execute:

```bash
bash gen_train.bash
``` 


## Visualization

In the output folder you will find a series of `.hdf5` containers. These can be visualized with the script:

```bash
blenderproc vis hdf5 output/0/*.hdf5
``` 
