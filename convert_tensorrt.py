from argparse import ArgumentParser
import torch
from mmdet2trt import mmdet2trt


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default='/workspace/odt/configs/ssd300_coco.py', help='mmdet Config file')
    parser.add_argument('--checkpoint', default='/workspace/odt/work_dirs/ssd300_coco/latest.pth', help='mmdet Checkpoint file')
    parser.add_argument('--save_path', default='/workspace/odt/save/ssd300_6epoch.pth', help='tensorrt model save path')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--fp16', type=bool, default=True, help='enable fp16 inference')
    args = parser.parse_args()

    cfg_path = args.config

    trt_model = mmdet2trt(
        cfg_path, args.checkpoint, fp16_mode=args.fp16, device=args.device)
    torch.save(trt_model.state_dict(), args.save_path)
    
if __name__ == '__main__':
    main()