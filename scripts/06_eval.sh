#python -m pdb tools/eval.py -f exps/retails/yolox_s.py -c YOLOX_outputs/yolox_s/best_ckpt.pth -n yolox-s -b 1 -d 1 --conf 0.001 --fp16 --fuse

python -m pdb tools/eval.py -f exps/retails/yolox_s.py -c YOLOX_outputs/yolox_s_diantou_300_fp16/best_ckpt.pth -n yolox-s -b 1 -d 1 --conf 0.001 --fp16 --fuse
