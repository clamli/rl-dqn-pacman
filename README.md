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
