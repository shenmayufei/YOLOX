#python3 tools/export_onnx.py -f exps/retails/yolox_s.py -c YOLOX_outputs/yolox_s/best_ckpt.pth -n yolox-s
#python3 -m pdb tools/export_onnx_test.py -f exps/retails/yolox_s.py -c YOLOX_outputs/yolox_s_diantou_300_fp16/best_ckpt.pth -n yolox-s-diantou300
#python3 -m pdb tools/export_onnx_test01.py -f exps/retails/yolox_s.py -c YOLOX_outputs/yolox_s_diantou_300_fp16/best_ckpt.pth -n yolox-s-diantou300  #ok for quat
#python3 -m pdb tools/export_onnx_test01.py -f exps/retails/yolox_s.py -c YOLOX_outputs/yolox_s/best_ckpt.pth -n yolox-s  #ok for yolox_s

python3 -m pdb tools/export_onnx_qat.py -f exps/retails/yolox_s.py -c YOLOX_outputs/yolox_s/best_ckpt.pth  #ok for quat

