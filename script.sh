#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python tools/train.py work_dirs/swin_tiny/Experience1/negative90/jml_improve.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py work_dirs/swin_tiny/Experience1/negative80/jml_improve.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py work_dirs/swin_tiny/Experience1/negative70/jml_improve.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py work_dirs/swin_tiny/Experience1/negative60/jml_improve.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py work_dirs/swin_tiny/Experience1/negative50/jml_improve.py
