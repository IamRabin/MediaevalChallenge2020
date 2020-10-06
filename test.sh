#!/bin/bash

do
if  python3 /content/polyp_seg/inference.py --root "/content/Kvasir-SEG"  --weights /content/weights/unet.pt --batch-size 16 ; then
    echo "Make sure you have tensorflow 2.0 or greater to open test results in tensorboard"
    exit
done
