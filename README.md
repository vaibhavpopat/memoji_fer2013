# memoji_fer2013
FER2013 challenge. 

## Architecture
Image -> conv2d -> (custom_resnet_block x2 + conv2d) x4 -> conv2d -> fc -> output

## Stats
Training accuracy after 30 epochs 98%
Test accuracy after 30 epochs 97.6%
