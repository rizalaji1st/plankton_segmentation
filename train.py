import segmentation_models_pytorch as smp

if __name__ == '__main__':
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=1,
        classes=3
    )