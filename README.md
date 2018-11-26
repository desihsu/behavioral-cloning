# Behavioral Cloning

Behavioral cloning using a [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) adapted from [NVIDIA's](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) architecture.

### Model Architecture

![alt text](./examples/architecture.png)

## Environment Setup

1. Download simulator  
  * [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)  
  * [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)  
  * [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)  
2. Clone this repo
3. ```~$ conda env create -f environment.yml```  
4. ```~$ source activate cloning```

## Usage

1. Run the simulator (Autonomous mode)
2. Connect: ```~$ python drive.py model.h5```

## Preview

![alt text](./examples/video_output.gif)