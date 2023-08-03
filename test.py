from pix2Topix import pix2pixG_256, pix2pixG_128
import torch
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import os
import glob
from tqdm import tqdm
import numpy as np

def test(img_path):
    if img_path.endswith('.png'):
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]
    else:
        img = Image.open(img_path)

    transforms = transform.Compose([
        transform.ToTensor(),
        #transform.Resize((128, 128)),
        transform.Resize((256, 256)),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transforms(img.copy())
    img = img[None].to('cuda')  # [1,3,128,128]

    # 实例化网络
    #G = pix2pixG_128().to('cuda')
    G = pix2pixG_256().to('cuda')
    # 加载预训练权重
    ckpt = torch.load('my_weights/256_16_100.pth')
    G.load_state_dict(ckpt['G_model'], strict=False)

    G.eval()
    out = G(img)[0]
    out = out.permute(1,2,0)
    out = (0.5 * (out + 1)).cpu().detach().numpy()
    return out
    # plt.figure()
    # plt.imshow(out)
    # plt.show()

def compare():
    test_path = './test'
    results_psnr = []
    sum_psnr = 0
    # results_ssim = []
    sum_ssim = 0
    test_img = glob.glob(test_path + '/*.png')
    for img in tqdm(test_img):
        ori_path = os.path.splitext(img)[0] + '.jpg'
        ori_img = cv2.imread(ori_path)
        #ori_img = cv2.resize(ori_img, (128, 128))
        ori_img = cv2.resize(ori_img, (256, 256))
        test_img = test(img)
        test_img = (test_img * 255).astype(np.uint8)
        result_psnr = compare_psnr(ori_img, test_img)
        sum_psnr += result_psnr
        results_psnr.append(result_psnr)
        result_ssim = compare_ssim(ori_img, test_img, channel_axis=-1)
        sum_ssim += result_ssim
        # results_ssim.append(result_ssim)
    pic_sum = len(results_psnr)
    mean_psnr = sum_psnr / pic_sum
    mean_ssim = sum_ssim / pic_sum
    return mean_psnr, mean_ssim


if __name__ == '__main__':
    img = test('./test/cmp_b0084.png')
    cv2.imshow('img', img)
    cv2.imwrite('cmp_b0084_out.jpg', img*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # mean1, mean2 = compare()
    # print('The average PSNR value is ' + str(mean1))
    # print('The average SSIM value is ' + str(mean2))
