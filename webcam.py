import os
import argparse
import torch
import time
import cv2
import numpy as np
import torchvision.transforms as transforms

from model import XLSR
from dataset import create_dataloader
from metric import cal_psnr
from visualization import save_res

def infer(transform, input_img):
    LR_img = transform(input_img)
    LR_img = LR_img[None]
    LR_img = LR_img.to(device).float()
    if device != 'cpu':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    HR_pred = model(LR_img)
    pred_img = HR_pred[0].cpu().numpy().transpose(1,2,0)
    if device != 'cpu':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    inference_time = t1 - t0
    return pred_img

def test(model, device):
    with torch.no_grad():
        print("warm up ...")
        random_input = torch.randn(1, 3, 640, 360).to(device)
        # warm up
        for _ in range(10):
            model(random_input)

        with torch.autograd.profiler.profile() as prof:
            model(random_input)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        transform = transforms.ToTensor()
        print("Start the inference ...")
        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            if ret:
                pred_img = infer(transform, frame)
                scaled = cv2.resize(pred_img, (1920, 1440))
                cv2.imshow("orig", frame)
                cv2.imshow("scale", scaled)
                cv2.imshow("sisr", pred_img)
                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                print("Cannot open img source")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str,
                        default='exp/OneCyclicLR', help='hyperparameters path')
    parser.add_argument('--SR-rate', type=int, default=3,
                        help='the scale rate for SR')
    parser.add_argument('--model', type=str, default='',
                        help='the path to the saved model')
    parser.add_argument('--device', type=str,
                        default='cpu', help='gpu id or "cpu"')
    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)

    # cuDnn configurations
    if opt.device != 'cpu':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # txt file to record training process
    # txt_path = os.path.join(opt.save_dir, 'test_res.txt')
    # if os.path.exists(txt_path):
    #     os.remove(txt_path)

    # folder to save the predicted HR image in the validation
    test_folder = os.path.join(opt.save_dir, 'test_res')
    os.makedirs(test_folder, exist_ok=True)

    device = 'cuda:' + str(opt.device) if opt.device != 'cpu' else 'cpu'
    model = XLSR(opt.SR_rate)

    # load pretrained model
    if opt.model.endswith('.pt') and os.path.exists(opt.model):
        model.load_state_dict(torch.load(opt.model, map_location=device))
    else:
        model.load_state_dict(torch.load(os.path.join(
            opt.save_dir, 'best.pt'), map_location=device))
    model.to(device)
    model.eval()

    # test_dataloader = create_dataloader('test', opt.SR_rate, False, batch_size=1, shuffle=False, num_workers=1)

    # evaluate
    test(model, device)

    # print("Saving the predicted HR images")
    # save_res(pred_list, name_list, test_folder)
    # print(f"Testing is done!, predicted HR images are saved in {test_folder}")
