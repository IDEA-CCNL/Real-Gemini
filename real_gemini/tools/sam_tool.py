import os
import wget
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
import uuid

from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator

from .utils import get_new_image_name


class Segmenting(object):
    _name_="Segment the Image"
    _description_="""当你想要对图片中的所有物体进行分割，但是不想对某个物体进行分割时很有用。比如：分割这张图片中的所有物体，或者在这张图片上生成分割结果，或者对这张图片进行分割，或者分割这张图片中的所有物体。输入到这个工具的应该是一个字符串，表示图片的路径或者图片所在的文件夹路径。
                useful when you want to segment all the part of the image, but not segment a certain object.like: segment all the object in this image, or generate segmentations on this image,
                or segment the image,"
                or perform segmentation on this image, "
                or segment all the object in this image."
                The input to this tool should be a string, representing the image_dir or image_paths."""
    
    def __init__(self, device):
        print(f"Inintializing Segmentation to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_checkpoint_path = os.path.join("checkpoints","sam")

        self.download_parameters()
        self.sam = build_sam(checkpoint=self.model_checkpoint_path).to(device)
        self.sam_predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        
        self.saved_points = []
        self.saved_labels = []

    def download_parameters(self):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        if not os.path.exists(self.model_checkpoint_path):
            wget.download(url,out=self.model_checkpoint_path)

        
    def show_mask(self, mask: np.ndarray,image: np.ndarray,
                random_color: bool = False, transparency=1) -> np.ndarray:
        
        """Visualize a mask on top of an image.
        Args:
            mask (np.ndarray): A 2D array of shape (H, W).
            image (np.ndarray): A 3D array of shape (H, W, 3).
            random_color (bool): Whether to use a random color for the mask.
        Outputs:
            np.ndarray: A 3D array of shape (H, W, 3) with the mask
            visualized on top of the image.
            transparenccy: the transparency of the segmentation mask
        """
        
        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255

        image = cv2.addWeighted(image, 0.7, mask_image.astype('uint8'), transparency, 0)


        return image

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
        ax.text(x0, y0, label)

    
    def get_mask_with_boxes(self, image_pil, image, boxes_filt):

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )
        return masks
    
    def segment_image_with_boxes(self, image_pil, image_path, boxes_filt, pred_phrases):

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image)

        masks = self.get_mask_with_boxes(image_pil, image, boxes_filt)

        # draw output image

        for mask in masks:
            image = self.show_mask(mask[0].cpu().numpy(), image, random_color=True, transparency=0.3)

        updated_image_path = get_new_image_name(image_path, func_name="segmentation")
        
        new_image = Image.fromarray(image)
        new_image.save(updated_image_path)

        return updated_image_path

    def set_image(self, img) -> None:
        """Set the image for the predictor."""
        with torch.cuda.amp.autocast():
            self.sam_predictor.set_image(img)

    def show_points(self, coords: np.ndarray, labels: np.ndarray,
                image: np.ndarray) -> np.ndarray:
        """Visualize points on top of an image.

        Args:
            coords (np.ndarray): A 2D array of shape (N, 2).
            labels (np.ndarray): A 1D array of shape (N,).
            image (np.ndarray): A 3D array of shape (H, W, 3).
        Returns:
            np.ndarray: A 3D array of shape (H, W, 3) with the points
            visualized on top of the image.
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        for p in pos_points:
            image = cv2.circle(
                image, p.astype(int), radius=3, color=(0, 255, 0), thickness=-1)
        for p in neg_points:
            image = cv2.circle(
                image, p.astype(int), radius=3, color=(255, 0, 0), thickness=-1)
        return image


    def segment_image_with_click(self, img, is_positive: bool,
                            evt: gr.SelectData):
                            
        self.sam_predictor.set_image(img)
        self.saved_points.append([evt.index[0], evt.index[1]])
        self.saved_labels.append(1 if is_positive else 0)
        input_point = np.array(self.saved_points)
        input_label = np.array(self.saved_labels)
        
        # Predict the mask
        with torch.cuda.amp.autocast():
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

        img = self.show_mask(masks[0], img, random_color=False, transparency=0.3)

        img = self.show_points(input_point, input_label, img)

        return img

    def segment_image_with_coordinate(self, img, is_positive: bool,
                            coordinate: tuple):
        '''
            Args:
                img (numpy.ndarray): the given image, shape: H x W x 3.
                is_positive: whether the click is positive, if want to add mask use True else False.
                coordinate: the position of the click
                          If the position is (x,y), means click at the x-th column and y-th row of the pixel matrix.
                          So x correspond to W, and y correspond to H.
            Output:
                img (PLI.Image.Image): the result image
                result_mask (numpy.ndarray): the result mask, shape: H x W

            Other parameters:
                transparency (float): the transparenccy of the mask
                                      to control he degree of transparency after the mask is superimposed.
                                      if transparency=1, then the masked part will be completely replaced with other colors.
        '''
        self.sam_predictor.set_image(img)
        self.saved_points.append([coordinate[0], coordinate[1]])
        self.saved_labels.append(1 if is_positive else 0)
        input_point = np.array(self.saved_points)
        input_label = np.array(self.saved_labels)

        # Predict the mask
        with torch.cuda.amp.autocast():
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )


        img = self.show_mask(masks[0], img, random_color=False, transparency=0.3)

        img = self.show_points(input_point, input_label, img)

        img = Image.fromarray(img)
        
        result_mask = masks[0]

        return img, result_mask

    def inference_all(self,image_input: str):
        if os.path.isdir(image_input):
            image_paths = [
                os.path.join(image_input, path) for path in os.listdir(image_input)
            ]
        else:
            image_paths = image_input.split(",")
        updated_image_paths = []
        for image_path in image_paths:
            if "segment-image" in image_path:
                continue
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = self.mask_generator.generate(image)
            plt.figure(figsize=(20,20))
            plt.imshow(image)
            if len(masks) == 0:
                return
            sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in sorted_anns:
                m = ann['segmentation']
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.random.random((1, 3)).tolist()[0]
                for i in range(3):
                    img[:,:,i] = color_mask[i]
                ax.imshow(np.dstack((img, m)))

            updated_image_path = get_new_image_name(image_path, func_name="segment-image")
            updated_image_paths.append(updated_image_path)
            plt.axis('off')
            plt.savefig(
                updated_image_path, 
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )
        return updated_image_paths