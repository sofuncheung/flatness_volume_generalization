# Example code for paper *WHY FLATNESS CORRELATES WITH GENERALIZATION FOR DEEPNEURAL NETWORKS*

The initial state of this example code can reproduce Fig. 3(c) and Fig. 3(f) in the paper.
For other plots we need to change the neural network architecture/dataset/optimizer manually. 

## Environment:

Platform: Linux
GPU: NVIDIA 2080 Ti

First create and **activate** your own virtual environment, then run: 
`python3 -m pip install -r requirements.txt`

## Run the main code

`python main.py -p $(pwd) -s 1`

`-p` indicates the path you want output files to be. `-s` means it is the first sample.

The main code does three things:
1. Train the DNN to zero error
2. calculate sharpness
3. calculate volume

If everything runs correctly, you will get a `record_1.npy`, which contains a numpy array
    [generalization accuracy, (log) sharpness, (log) volume].
    You can then use whatever tool you like to plot them.

### Change the `attack_set_size`

In order to get functions with various generalization performance, we can manually tune the `attack_set_size` in `config.py`. The larger `attack_set_size` is (compared to the uncorrupted training set), the worse the generalization. 

Here is an example of bash script that submit multiple jobs:

`
for i in `seq 0 50 500`
do
    echo "Attack size: $i"
    mkdir attack_size_$i
    cp config.py attack_size_$i/
    cd attack_size_$i/
    echo "    attack_set_size = $i" >> config.py
    dir=$(pwd)
    echo "$dir"

    for s in \`seq 1 1 5\`
    do
        echo "submitting sample $s"
        python main.py -p $dir -s $s
    done
    cd ..
done
`
**Note:** if you open new directories to run the main code (e.g. create directories like `attack_size_0` and contain `config.py` in these individual directories) then you need to **remove** the `config.py` in the *same directory* as `main.py`, or the code will take that very one as the configs.

