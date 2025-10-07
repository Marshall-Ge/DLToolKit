## Image Classification
Run a simple MNIST classification model with default settings:
```bash
python -m dltoolkit.trainer.img_cls model.name_or_path=CovNet_MNIST data.name_or_path=ylecun/mnist
```
Use a ResNet model for CIFAR-10 classification:

```bash
python -m dltoolkit.trainer.img_cls \
    model.name_or_path=ResNet18 \
    +model.param.expansion=1 \
    data.name_or_path=uoft-cs/cifar10 \
    data.image_key=img \
    data.eval_dataset=uoft-cs/cifar10 \
    data.eval_split=test \
    data.num_classes=10 \
    trainer.max_epochs=10 \
    trainer.learning_rate=0.001
```
## Language Model Fine-tuning

Fine-tune a small language model:
```bash
python -m dltoolkit.trainer.lm \
    model.name_or_path=facebook/opt-125m \
    data.name_or_path=Self-GRIT/wikitext-2-raw-v1-preprocessed \
    data.split=train \
    trainer.train_batch_size=32 \
    trainer.max_epochs=3 \
    data.max_len=512 \
    strategy.mixed_precision=fp16
```

## Continual Learning

Run a continual learning experiment on CIFAR-10:

```bash
python -m dltoolkit.trainer.cl_trainer.img_cls \
    model.name_or_path=incResNet18 \
    data.name_or_path=uoft-cs/cifar10 \
    trainer.init_cls=2 \
    trainer.increment=2 \
    trainer.capacity=5000
```

## Others
Run with wandb logging:
```bash
python -m dltoolkit.trainer.img_cls \
    model.name_or_path=CovNet_MNIST \
    data.name_or_path=ylecun/mnist \
    trainer.max_epochs=10 \
    tracker.use_wandb={your_wandb_api_key} \
    tracker.wandb_project=dltoolkit_demo \
    tracker.wandb_run_name=img_cls \
#    tracker.wandb_group=img_cls_experiments \
#    tracker.wandb_entity={your_wandb_entity}
```