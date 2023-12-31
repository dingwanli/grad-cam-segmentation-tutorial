import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #对应 卡1
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
import cv2
import torch
import numpy as np
from matplotlib import pylab as plt

from Network.stage_DTM_flexibility import Stage_DTM
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU
print('GPU num: ', torch.cuda.device_count())

def draw_fig(img_mat, title='img_mat'):
    plt.close()
    plt.title(title)
    plt.imshow(img_mat, cmap='gray')
    # plt.imshow(img_mat, cmap='binary')
    plt.show()
    plt.close()

def CAM_seg_1folder(model, image_npy_path, cam_png_path, target_layer, category='endocardium'):
    sem_classes = ['__background__', 'endocardium', 'epicardium']
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    category_category = sem_class_to_idx[category]
    for image_sub_filename in os.listdir(image_npy_path):
        cam_png_filename = os.path.splitext(image_sub_filename)[0] + '.png'
        grayscale_npy_filename = os.path.splitext(image_sub_filename)[0] + '.npy'
        image_sub_filepath = os.path.join(image_npy_path, image_sub_filename)
        image_mat = np.load(image_sub_filepath)

        image_mat = np.array(image_mat, dtype=np.float32)
        image_mat = image_mat / 255
        image_mat = np.squeeze(image_mat)
        image_mat = image_mat[..., np.newaxis]
        image_mat = cv2.resize(image_mat, (128, 128))
        raw_image_mat = image_mat[..., np.newaxis]
        image_mat = image_mat[np.newaxis, ...]
        image_mat = image_mat[np.newaxis, ...]
        input_tensor = torch.from_numpy(image_mat)
        input_tensor = input_tensor.cuda()

        output = model(input_tensor)  # torch.Size([1, 3, 128, 128])
        normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()

        category_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
        category_mask_float = np.float32(category_mask == category_category)

        class SemanticSegmentationTarget:
            def __init__(self, category, mask):
                self.category = category
                self.mask = torch.from_numpy(mask)
                if torch.cuda.is_available():
                    self.mask = self.mask.cuda()
            def __call__(self, model_output):
                return (model_output[self.category, :, :] * self.mask).sum()

        targets = [SemanticSegmentationTarget(category_category, category_mask_float)]
        with GradCAM(model=model, target_layers=target_layer, use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(raw_image_mat, grayscale_cam)
            # cam_image = show_cam_on_image(raw_img, grayscale_cam, use_rgb=True)
        cv2.imwrite(cam_png_path + cam_png_filename, cam_image)
    print('finish!!!')

if __name__ in "__main__":

    model = Stage_DTM(num_class=3)
    model = model.to(device)
    # model = nn.DataParallel(model)
    state_load_path = '/data/dwl/all_age_data/model/train_DTM_flexibility_all/paper/train_DTM_flexibility_all_mc_5.pth'
    model.load_state_dict(torch.load(state_load_path), strict=False)
    model.load_state_dict(torch.load(state_load_path))
    model = model.eval()
    # for (name, module) in model.named_modules():
    #     print(name)
    target_layer = [model.decoder.pred_norm]
    image_npy_path = '/data/dwl/all_age_data/model/train_DTM_all/paper/train_DTM_all_mc_5/CAM/input_adjust/'
    cam_png_path = '/data/dwl/all_age_data/model/demo/output_background_CAM/'
    CAM_seg_1folder(model, image_npy_path, cam_png_path, target_layer, category='__background__')

