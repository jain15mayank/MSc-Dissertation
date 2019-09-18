The code is organized into 2 parent directories:
    1. Final Code - Contain final code for computer graphics scheme and PSO search framework
    2. Models - Contain trained models and training files

The VGG-16 model will be placed in 'VGG16 - Data Random - LR 1e-7' directory inside the 'Models' directory after the training. The trained model is not included due to its significantly large size.
The traing file to setup the baseline is 'vgg16_train.py' present inside the 'Models' directory too.

Main code for PSO search framework is 'adversarialAgent.py', present in the 'Final Code' folder.
'pso_pyswarm_parallel.py' is the parallelized and vectorized version of PySwarm library's source code, optimized for the current task.
'utils_mudSlap.py' and 'utils_naturalPerturbations.py' are the utility files for Computer Graphics Scheme.

Another sub-directory titled 'Real World Experiment' (present inside 'Final Code') contains Image Data from the real world experiment and another version of 'adversarialAgent.py', written to parse that Image Data.

Note: GTSRB Data must be parsed and saved in a directory hierarchy in accordance to its use in the following files:
    - Final Code/adversarialAgent.py
    - Models/vgg16_train.py