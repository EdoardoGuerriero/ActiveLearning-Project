# Building task-specific stance identification models: an evaluation of the active learning approach

Reposetory containing the code for the paper. All code is in python3, for the CNN architecture we used Pytorch. 
The final version of the paper can be found in the 'paper' folder. 

## Files description:

**main.py** - contains the main function to run one or several active learning training simulations <br />
**active_learning.py** - contains the implementation of the active learning selection strategies <br />
**model_functions.py** - contains the main model architecture, plus all extra functions required (train, evaluate, extra metrics, etc.) <br />
**simulation.py** - contains the main function of the active learning simulation. <br />
**utils.py** - contains extra functions required to save/load pytorch fields and iterators (normally not useful, but necessary for extra analysis of our simulations)  <br />
**temperature_scaling.py** - contains the main function to perform the post processing of the model predictions. Taken from the authors' [repository](https://github.com/gpleiss/temperature_scaling) of the original paper. 
