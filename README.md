# Computer Vision Final Project: CycleGANs

## How to run
Sample VS Code debug file:
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/code/run.py",
            "console": "integratedTerminal",
            "args": ["--dataset", "horse-zebra"],
            "cwd": "${fileDirname}"
        }
    ]
}
```
## Data
Data can be found in the data directory and downloaded with the `get_data.sh` script.

## Checkpoints
Pre-trained checkpoints can be found in [this google drive](https://drive.google.com/drive/folders/1fM165n4-7gPV_xVI_vxJ8Kajiym0XFK5?usp=sharing)