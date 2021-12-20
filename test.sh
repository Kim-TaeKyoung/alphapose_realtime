cd $ALPHAPOSE_DIR
python3 scripts/demo_inference.py --detector tracker \
                                  --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
                                  --checkpoint pretrained_models/fast_res50_256x192.pth \
                                  --webcam rtsp://ADMIN:9105@211.117.64.9:5554/live/main1 \
                                  --save_video