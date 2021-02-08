# Example code usage:


## Environment:

Platform: Linux
GPU: NVIDIA 2080 Ti

First create and **activate** your own virtual environment, then run: 
`python3 -m pip install -r requirements.txt`

## Run the main code

`python main.py -p $(pwd)`
It does three things:
1. Train the DNN to zero error
2. calculate sharpness
3. calculate volume

### Change the `attack_set_size` in config.py manually
