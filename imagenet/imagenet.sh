
# train imagenet model, baseline or openmix
python -m torch.distributed.launch --nproc_per_node=6 main_imagenet.py --arch resnet50 --method OpenMix --b 256 --workers 16 --opt-level O1 ./

# inference
python -m torch.distributed.launch --nproc_per_node=1 test_imagenet.py --b 256 --workers 16 --opt-level O1 ./