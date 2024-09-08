The main script is `train.py`. It trains the models based on the chosen configuration. Scripts `launch_ffno.sh` and `launch_mifno.sh` shows how to define the variables to train the models.

Before training, you should create folders `logs/loss` and `logs/models` at the directory root to store the loss functions and the trained models.

Multi-GPU training is implemented with horovod. It can be changed depending on your computing centre. Same for the `launch_ffno.sh` and `launch_mifno.sh`, adapt the syntax to your computing environment.
