# Example code usage:


## Environment:

Platform: Linux
GPU: NVIDIA 2080 Ti

First create and **activate** your own virtual environment, then run: 
`python3 -m pip install -r requirements.txt`

## Run the main code

`python main.py -p $(pwd) -s 1`

-p indicates the path you want output files to be. -s means it is the first sample.

The main code does three things:
1. Train the DNN to zero error
2. calculate sharpness
3. calculate volume

if everything runs correctly, you will get a `record_1.npy`, which contains a numpy array
    (generalization accuracy, $log_{10}(sharpness)$, $log_{10}V(f)$).
    You can then use whatever tool you like to plot them.

### Change the `attack_set_size` in config.py manually
