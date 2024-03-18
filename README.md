# Heatmap-Perimeter-Identification

## CC virtual environment setup
1. module load StdEnv/2023
2. module load opencv/4.9.0
3. virtualenv --no-download venv_py311
4. source venv_py311/bin/activate
5. pip install --no-index --upgrade pip
6. pip install --no-index numpy
7. pip install --no-index torch torchvision
8. pip install --no-index tensorboard
9. pip install --no-index gymnasium

## TODO
1. Model save and load
2. RB save and load
3. reward plot at the end
4. Tensorboard - add gradients
5. new image with less circles
6. __do not allow less than 3 circles__


Load tensorboard 
in batch script:
```bash
tensorboard --logdir=runs --host 0.0.0.0 --load_fast false &
```
in local shell:
```shell
ssh -N -f -L localhost:6006:<node from sq>:6006 ataitler@cedar.computecanada.ca
```