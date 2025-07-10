# custom-unet-pytorch

custom-unet-pytorch/

This repo contains a modified UNet model with:

- Transposed convolutions for upsampling
- LeakyReLU activations
- Residual-style bottleneck
- Dropout (can be added in decoder for regularization)

Useful for segmentation and generation tasks where spatial accuracy is important.
