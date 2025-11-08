python -m dltoolkit.trainer.img_cls \
    model.name_or_path=ResNet18 \
    +model.param.expansion=1 \
    data.name_or_path=uoft-cs/cifar10 \
    data.eval_dataset=uoft-cs/cifar10 \
    model.param.in_channel=3 \
    data.image_key=img \
    data.eval_split=test \
    trainer.max_epochs=10 \
    trainer.learning_rate=0.001 \
    save_path=./outputs/cifar10_resnet18_1/



python -m dltoolkit.trainer.img_cls \
    model.name_or_path=ResNet18 \
    +model.param.expansion=1 \
    data.name_or_path=ylecun/mnist \
    model.param.in_channel=1 \
    trainer.max_epochs=10 \
    trainer.learning_rate=0.001 \
    save_path=./outputs/mnist_resnet18_1/