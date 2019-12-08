## Requirements
- Python 3.6.3
- Pytorch 1.3
- Pygame 1.9.6
- CUDA (optional)

## Run the Code
Please first check out the config.py file to configure the settings for the experiment.

To train the model, run 
```
python dqn_train.py
```

Then you will get all the models and training results in a folder with the prefix of "model_dqn"

To test the model, modify the name of the model loaded in dqn_test.py, then run:
```
python dqn_test.py
```


## Config
```python
# True if on TACC
on_TACC = False 

# True if use CUDA
use_cuda = False 

# True if use low resolution(80x80), greyscale picture as the input
use_simple = False 

# True if use prioritized experience replay
use_per = False 

# True if use double DQN
use_double_dqn = False 
```
