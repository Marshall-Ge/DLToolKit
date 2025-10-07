## DLToolkit

A Toolkit for Deep Learning Projects. It provides a set of utilities and modules to streamline the development and deployment of deep learning models.

<img src="Framework.png" alt="Framework">

## Quick Start

Env Setup:
```bash
conda create -n dltoolkit python=3.10 -y
conda activate dltoolkit
pip install -e .
```
Run Example:
```bash
# detailed configs can be found in dltoolkit/trainer/config/img_cls.yaml
python -m dltoolkit.trainer.img_cls model.name_or_path=CovNet_MNIST data.name_or_path=ylecun/mnist trainer.max_epochs=5 
```

You can find output model files and checkpoints in './ckpt' as default.

By default, we use tensorboard as logs tracker(wandb also supported) and you can find training logs in './logs' as default, you can visualize them by:
```bash
tensorboard --logdir=./logs
```

You can find more usage example in [EXAMPLES](example.md), have fun!

## How to Use & Develop
Normally, you should use this toolkit as a template, you need to customize your own dataset, model but rather than focus on the training and evaluation process. Below 
is the recommend steps to use this toolkit:

1. Confirm your env is ready, and install the dependencies by `pip install -e .`
2. Confirm your training task, for example, image classification, text classification, text generation, etc. You can find tasks supported
in 'dltoolkit/trainer' folder, and theses files are the entry points of your training and evaluation process.
3. Specify your dataset, now we only support datasets from Huggingface. For example, 'data.name_or_path=ylecun/mnist', more datasets related configs can be found in 'dltoolkit/trainer/config' folder.
4. If you are using a model from Huggingface, Then you just need to specify the model name or path in the config file, like 'model.name_or_path=bert-base-uncased'.
If you would like to customize your own model, you can create a new file in 'dltoolkit/models' folder, and implement your model class by inheriting 'torch.nn.Module'. Then you
need to register it by adding a decorator `@MODELS.register_module()`. For example:
```python
from dltoolkit.utils.utils import MODEL_REGISTRY
import torch.nn as nn
import torch.nn.functional as F

@MODEL_REGISTRY.register()
class CovNet_MNIST(nn.Module):
    def __init__(self, config):
        super(CovNet_MNIST, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
```
5. Finally, you should run 'precommit.sh' to do some code modification, now you can run your training task by:
```bash
python -m dltoolkit.trainer.<your_task> <configs>
```
6. Enjoy it!


## References & Acknowledgements

Thanks the following wonderful works, this project draws inspirations from them:
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [DeepSpeedChat](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
