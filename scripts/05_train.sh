#export YOLOX_DATADIR=/data/yzh/data/diantou/train_20210807_train
#ln -s /data/yzh/data/diantou/train_20210807_train ./datasets/diantou


#python tools/train.py -f exps/retails/yolox_s.py -d 2 -b 16 --fp16 --resume
export YOLOX_DATADIR=/data/yzh/data/bottle/20211001_train
export CUDA_VISIBLE_DEVICES=1
python  tools/train.py -f exps/retails/yolox_s_bottle.py -d 1 -b 16 --fp16 --resume
