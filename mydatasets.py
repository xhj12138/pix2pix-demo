from torch.utils.data.dataset import Dataset
import torchvision.transforms as transform
from PIL import Image
import cv2


class CreateDatasets(Dataset):
    def __init__(self, ori_imglist,img_size):
        self.ori_imglist = ori_imglist
        self.transform = transform.Compose([
            transform.ToTensor(),
            transform.Resize((img_size, img_size)),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.ori_imglist)

    def __getitem__(self, item):
        ori_img = cv2.imread(self.ori_imglist[item])
        ori_img = ori_img[:, :, ::-1]
        real_img = Image.open(self.ori_imglist[item].replace('.png', '.jpg'))
        ori_img = self.transform(ori_img.copy())
        real_img = self.transform(real_img)
        return ori_img, real_img
