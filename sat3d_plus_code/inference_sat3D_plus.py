import gc
import os
import os.path as osp
import json
import pickle
from glob import glob
from collections import OrderedDict
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import torchio as tio

from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from networks import Discriminator
from segment_anything_with_swin_conf_plus.build_samswin3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D

from utils.click_method import (
    get_next_click3D_torch_ritm,
    get_next_click3D_torch_2,
    get_next_click3D_torch_with_dice_rev,
)
from utils.data_loader_tumors_scalability_text_val import Dataset_Union_ALL
from utils.tumor_data_paths_full_dataset_scalability_text import img_datas, all_datasets

join = osp.join


parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--checkpoint_path', type=str,
                    default='./work_dir/sat3D_4_plus/sam_model_dice_best.pth')
parser.add_argument('-ccp', '--critic_checkpoint_path', type=str,
                    default='./work_dir/sat3D_4_plus/critic_latest.pth')
parser.add_argument('--output_dir', type=str, default='./work_dir/sat3D_4_scalability_biomedclip/visualization_20')
parser.add_argument('--task_name', type=str, default='gtvp',
                    help='gtvp, gtvn, colon_cancer_primaries, edema, enhancing_tumor, hepatic_tumor, kidney_tumor, liver_tumor, lung_cancer, pancreas_cancer, non_enhancing_tumor, renal_tumor, kidney_tumor, tumor breast_tumor')
parser.add_argument('--dataset_id', type=int, default=0,
                    help='0: Autopet_ct, 1: Autopet_pet, 2: BraTS_2021_mr_t1, 3: BraTS_2021_mr_t2, 4: BraTS_2021_mr_flair, 5: BraTS_2021_mr_t1ce, 6: HNTSMRG24_mr_t2, 7: Task06_Lung_ct, 8: Task08_HepaticVessel_ct, 9: Task03_Liver_ct, 10: Task07_Pancreas_ct, 11: Task10_Colon_ct, 12: KiPA22, 13: KiTS23, 14: TDSC_ABUS')

parser.add_argument('--skip_existing_pred', action='store_true', default=False)
parser.add_argument('--save_image_and_gt', action='store_true', default=False)
parser.add_argument('--sliding_window', action='store_true', default=False)

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-mt', '--model_type', type=str, default='swin2')
parser.add_argument('-nc', '--num_clicks', type=int, default=20)
parser.add_argument('-pm', '--point_method', type=str, default='default')
parser.add_argument('-dt', '--data_type', type=str, default='Ts')

parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--ft2d', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=2023)

parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--max_scribbles', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=0)

args = parser.parse_args()

args.output_dir = join(args.output_dir, args.task_name)
args.pred_output_dir = join(args.output_dir, "pred")
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.pred_output_dir, exist_ok=True)

infer_task_name = [all_datasets[args.dataset_id]]
infer_img_data_path = [img_datas[args.dataset_id]]

args.save_name = join(args.output_dir, f"{args.task_name}_{infer_task_name[0]}_dice.py")
args.file_save_name = args.task_name + "_" + infer_task_name[0]
output_file = join(args.output_dir, f"{args.file_save_name}.txt")

print("output_dir set to", args.output_dir)

SEED = args.seed
print("set seed as", SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.init()

click_methods = {
    'default': get_next_click3D_torch_with_dice_rev,
    'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch_2,
}


def compute_iou(pred_mask, gt_semantic_seg):
    in_mask = np.logical_and(gt_semantic_seg, pred_mask)
    out_mask = np.logical_or(gt_semantic_seg, pred_mask)
    denom = np.sum(out_mask)
    if denom == 0:
        return np.nan
    return np.sum(in_mask) / denom


def compute_dice(mask_gt, mask_pred, dtype=np.uint8):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.nan
    volume_intersect = (mask_gt.astype(dtype) & mask_pred.astype(dtype)).sum()
    return 2 * volume_intersect / volume_sum


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict


def save_numpy_to_nifti(in_arr: np.array, out_path, ref_seg_img):
    ori_arr = np.transpose(in_arr.squeeze(), (2, 1, 0))
    out = sitk.GetImageFromArray(ori_arr)
    out.CopyInformation(ref_seg_img)
    sitk.WriteImage(out, out_path)


def save_numpy_to_nifti_hecktor_pet(in_arr: np.array, out_path, ref_seg_img):
    ori_arr = np.transpose(in_arr.squeeze(), (2, 1, 0))
    out = sitk.GetImageFromArray(ori_arr)
    sitk.WriteImage(out, out_path)


def save_numpy_to_nifti_autopet(in_arr: np.array, out_path, ref_seg_img):
    ori_arr = np.transpose(in_arr.squeeze(), (2, 1, 0))
    out = sitk.GetImageFromArray(ori_arr)
    out.CopyInformation(ref_seg_img)
    sitk.WriteImage(out, out_path)


def save_numpy_to_nifti_kits(in_arr: np.array, out_path, ref_seg_img):
    ori_arr = in_arr.squeeze()
    out = sitk.GetImageFromArray(ori_arr)
    out.CopyInformation(ref_seg_img)
    sitk.WriteImage(out, out_path)


def save_case_array(in_arr, out_path, ref_seg_img, dataset_id):
    if dataset_id in [0, 1]:
        save_numpy_to_nifti_autopet(in_arr, out_path, ref_seg_img)
    elif dataset_id == 13:
        save_numpy_to_nifti_kits(in_arr, out_path, ref_seg_img)
    elif dataset_id == 20:
        save_numpy_to_nifti_hecktor_pet(in_arr, out_path, ref_seg_img)
    else:
        save_numpy_to_nifti(in_arr, out_path, ref_seg_img)


def overlay_points_on_pred(pred3D_full, points, upto_idx, radius=2, value=10):
    pred_vis = pred3D_full.copy()
    for pt in points[:upto_idx + 1]:
        pt_np = np.array(pt)
        if pt_np.ndim == 3:
            coords_list = pt_np[0]
        elif pt_np.ndim == 2:
            coords_list = pt_np
        else:
            continue

        for coord in coords_list:
            x, y, z = [int(round(c)) for c in coord]
            x0 = max(0, x - radius)
            x1 = min(pred_vis.shape[-3], x + radius)
            y0 = max(0, y - radius)
            y1 = min(pred_vis.shape[-2], y + radius)
            z0 = max(0, z - radius)
            z1 = min(pred_vis.shape[-1], z + radius)
            pred_vis[..., x0:x1, y0:y1, z0:z1] = value
    return pred_vis


def sample_positive_scribbles(gt3D, max_scribbles=10):
    """
    gt3D: (B, 1, D, H, W)
    returns:
        scribble_points: (B, N, 3) in (x, y, z)
        scribble_labels: (B, N) with valid=1, padded=-1
    """
    batch_points = []
    batch_labels = []

    B = gt3D.shape[0]
    device = gt3D.device

    for b in range(B):
        fg_coords = torch.nonzero(gt3D[b, 0] > 0, as_tuple=False).float()  # (z, y, x)

        if fg_coords.numel() == 0:
            pts = torch.zeros((0, 3), device=device, dtype=torch.float32)
            lbs = torch.zeros((0,), device=device, dtype=torch.long)
        else:
            num_pts = min(max_scribbles, fg_coords.shape[0])
            rand_idx = torch.randperm(fg_coords.shape[0], device=device)[:num_pts]
            pts = fg_coords[rand_idx][:, [2, 1, 0]]  # -> (x, y, z)
            lbs = torch.ones((num_pts,), device=device, dtype=torch.long)

        batch_points.append(pts)
        batch_labels.append(lbs)

    max_pts = max(p.shape[0] for p in batch_points)

    if max_pts == 0:
        scribble_points = torch.zeros((B, 0, 3), device=device, dtype=torch.float32)
        scribble_labels = torch.zeros((B, 0), device=device, dtype=torch.long)
    else:
        padded_points = []
        padded_labels = []
        for pts, lbs in zip(batch_points, batch_labels):
            n = pts.shape[0]
            if n < max_pts:
                pad_pts = torch.zeros((max_pts - n, 3), device=device, dtype=pts.dtype)
                pad_lbs = -torch.ones((max_pts - n,), device=device, dtype=torch.long)
                pts = torch.cat([pts, pad_pts], dim=0)
                lbs = torch.cat([lbs, pad_lbs], dim=0)

            padded_points.append(pts)
            padded_labels.append(lbs)

        scribble_points = torch.stack(padded_points, dim=0)
        scribble_labels = torch.stack(padded_labels, dim=0)

    return scribble_points, scribble_labels


def ensure_text_list(texts):
    if texts is None:
        return None
    if isinstance(texts, str):
        return [texts]
    if isinstance(texts, tuple):
        return list(texts)
    return texts


def batch_forward(
    sam_model,
    image_embedding,
    gt3D,
    low_res_masks,
    low_res_conf,
    points=None,
    boxes=None,
    text=None,
):
    device = image_embedding.device

    if low_res_conf is not None:
        low_res_conf = low_res_conf.to(device)

    if points is not None:
        coords, labels = points
        coords = coords.to(device)
        labels = labels.to(device)
        points = (coords, labels)

    if boxes is not None:
        boxes = boxes.to(device)

    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=points,
        boxes=boxes,
        masks=low_res_masks.to(device),
        conf=low_res_conf,
        text=text,
    )

    low_res_masks, _ = sam_model.mask_decoder(
        image_embeddings=image_embedding.float().to(device),
        image_pe=sam_model.prompt_encoder.get_dense_pe().float(),
        sparse_prompt_embeddings=sparse_embeddings.float(),
        dense_prompt_embeddings=dense_embeddings.float(),
        multimask_output=False,
    )

    prev_masks = F.interpolate(
        low_res_masks,
        size=gt3D.shape[-3:],
        mode='trilinear',
        align_corners=False
    )
    return low_res_masks, prev_masks


def get_points(prev_masks, gt3D, click_points, click_labels, click_method_name='random', multi_click=False):
    points, labels = click_methods[click_method_name](prev_masks, gt3D)

    if isinstance(points, list):
        if points[0].dim() == 3:
            points = torch.stack([p.squeeze(0) for p in points], dim=0)
            labels = torch.stack([l.squeeze(0) for l in labels], dim=0)
        elif points[0].dim() == 2:
            points = torch.stack(points, dim=0)
            labels = torch.stack(labels, dim=0)
        else:
            raise ValueError(f"Unexpected point dimension: {points[0].dim()}")
    else:
        if points.dim() == 4:
            points = points.squeeze(1)
            labels = labels.squeeze(1)
        elif points.dim() != 3:
            raise ValueError(f"Points tensor has unexpected dimension: {points.dim()}")

    click_points.append(points)
    click_labels.append(labels)

    points_multi = torch.cat(click_points, dim=1)
    labels_multi = torch.cat(click_labels, dim=1)

    if multi_click:
        return points_multi, labels_multi
    return points, labels


def finetune_model_predict3D(
    img3D,
    gt3D,
    boxes,
    texts,
    sam_model_tune,
    critic,
    device='cuda',
    click_method='random',
    num_clicks=10,
    max_scribbles=10,
    multi_click=False,
):
    torch.cuda.empty_cache()

    img3D = norm_transform(img3D.squeeze(dim=1))
    img3D = img3D.unsqueeze(dim=1)
    img3D = img3D.to(device)
    gt3D = gt3D.to(device).type(torch.long)

    texts = ensure_text_list(texts)

    if boxes is not None:
        boxes = boxes.to(device)
        if boxes.dim() == 2 and boxes.shape[-1] == 6:
            boxes = boxes.view(-1, 2, 3)
    else:
        boxes = None

    click_points = []
    click_labels = []
    pred_list = []

    scribble_points, scribble_labels = sample_positive_scribbles(gt3D, max_scribbles=max_scribbles)
    if scribble_points.numel() > 0:
        click_points.append(scribble_points)
        click_labels.append(scribble_labels)
        initial_points = (scribble_points, scribble_labels)
    else:
        initial_points = None

    prev_masks = torch.zeros_like(gt3D).to(device)
    low_res_masks = F.interpolate(
        prev_masks.float(),
        size=(args.crop_size // 4, args.crop_size // 4, args.crop_size // 4)
    )

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(img3D)

        zero_conf = torch.zeros_like(low_res_masks).to(device)

        low_res_masks, prev_masks = batch_forward(
            sam_model=sam_model_tune,
            image_embedding=image_embedding,
            gt3D=gt3D,
            low_res_masks=low_res_masks,
            low_res_conf=zero_conf,
            points=initial_points,
            boxes=boxes,
            text=texts,
        )

        medsam_seg_prob = torch.sigmoid(prev_masks)
        medsam_seg = (medsam_seg_prob.cpu().numpy().squeeze() > 0.5).astype(np.uint8)
        pred_list.append(medsam_seg)

        for click_idx in range(num_clicks):
            current_click_method = click_method if click_idx <= 1 else "default"

            points_input, labels_input = get_points(
                prev_masks=prev_masks,
                gt3D=gt3D,
                click_points=click_points,
                click_labels=click_labels,
                click_method_name=current_click_method,
                multi_click=multi_click,
            )

            conf_map = (torch.sigmoid(critic(torch.sigmoid(prev_masks).float())).to(device) > 0.5).float()
            low_res_conf = F.interpolate(
                conf_map.float(),
                size=(args.img_size // 4, args.img_size // 4, args.img_size // 4)
            )

            low_res_masks, prev_masks = batch_forward(
                sam_model=sam_model_tune,
                image_embedding=image_embedding,
                gt3D=gt3D,
                low_res_masks=low_res_masks,
                low_res_conf=low_res_conf,
                points=(points_input, labels_input),
                boxes=boxes,
                text=texts,
            )

            medsam_seg_prob = torch.sigmoid(prev_masks)
            medsam_seg = (medsam_seg_prob.cpu().numpy().squeeze() > 0.5).astype(np.uint8)
            pred_list.append(medsam_seg)

    del medsam_seg_prob, medsam_seg, prev_masks, img3D, gt3D
    gc.collect()
    torch.cuda.empty_cache()

    return pred_list, click_points, click_labels, scribble_points, scribble_labels


def pad_and_crop_with_sliding_window(img3D, gt3D, boxes, crop_transform, offset_mode="center"):
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=img3D.squeeze(0)),
        label=tio.LabelMap(tensor=gt3D.squeeze(0)),
    )
    padding_params, cropping_params = crop_transform.compute_crop_or_pad(subject)

    if cropping_params is None:
        cropping_params = (0, 0, 0, 0, 0, 0)
    if padding_params is None:
        padding_params = (0, 0, 0, 0, 0, 0)

    roi_shape = crop_transform.target_shape
    vol_bound = (0, img3D.shape[2], 0, img3D.shape[3], 0, img3D.shape[4])

    center_oob_ori_roi = (
        cropping_params[0] - padding_params[0], cropping_params[0] + roi_shape[0] - padding_params[0],
        cropping_params[2] - padding_params[2], cropping_params[2] + roi_shape[1] - padding_params[2],
        cropping_params[4] - padding_params[4], cropping_params[4] + roi_shape[2] - padding_params[4],
    )

    window_list = []
    offset_dict = {
        "rounded": list(product((-32, +32, 0), repeat=3)),
        "center": [(0, 0, 0)],
    }

    for offset in offset_dict[offset_mode]:
        oob_ori_roi = (
            center_oob_ori_roi[0] + offset[0], center_oob_ori_roi[1] + offset[0],
            center_oob_ori_roi[2] + offset[1], center_oob_ori_roi[3] + offset[1],
            center_oob_ori_roi[4] + offset[2], center_oob_ori_roi[5] + offset[2],
        )

        padding_params = [0 for _ in range(6)]
        for idx, (ori_pos, bound) in enumerate(zip(oob_ori_roi, vol_bound)):
            pad_val = 0
            if idx % 2 == 0 and ori_pos < bound:
                pad_val = bound - ori_pos
            if idx % 2 == 1 and ori_pos > bound:
                pad_val = ori_pos - bound
            padding_params[idx] = pad_val

        cropping_params = (
            oob_ori_roi[0] + padding_params[0], vol_bound[1] - oob_ori_roi[1] + padding_params[1],
            oob_ori_roi[2] + padding_params[2], vol_bound[3] - oob_ori_roi[3] + padding_params[3],
            oob_ori_roi[4] + padding_params[4], vol_bound[5] - oob_ori_roi[5] + padding_params[5],
        )

        pad_and_crop = tio.Compose([
            tio.Pad(padding_params, padding_mode=crop_transform.padding_mode),
            tio.Crop(cropping_params),
        ])
        subject_roi = pad_and_crop(subject)

        img3D_roi = subject_roi.image.data.clone().detach().unsqueeze(1)
        gt3D_roi = subject_roi.label.data.clone().detach().unsqueeze(1)

        # recompute bbox from ROI GT to keep prompt aligned with cropped volume
        fg_coords = torch.nonzero(gt3D_roi[0, 0] > 0, as_tuple=False).float()
        if fg_coords.numel() > 0:
            fg_coords = fg_coords[:, [2, 1, 0]]
            x_min, y_min, z_min = fg_coords.min(dim=0)[0]
            x_max, y_max, z_max = fg_coords.max(dim=0)[0]
            box_roi = torch.tensor([[x_min, y_min, z_min, x_max, y_max, z_max]], dtype=torch.float32)
        else:
            box_roi = torch.zeros((1, 6), dtype=torch.float32)

        windows_clip = [0 for _ in range(6)]
        for i in range(3):
            if offset[i] < 0:
                windows_clip[2 * i] = 0
                windows_clip[2 * i + 1] = -(roi_shape[i] + offset[i])
            elif offset[i] > 0:
                windows_clip[2 * i] = roi_shape[i] - offset[i]
                windows_clip[2 * i + 1] = 0

        pos3D_roi = dict(
            padding_params=padding_params,
            cropping_params=cropping_params,
            ori_roi=(
                cropping_params[0] + windows_clip[0],
                cropping_params[0] + roi_shape[0] - padding_params[0] - padding_params[1] + windows_clip[1],
                cropping_params[2] + windows_clip[2],
                cropping_params[2] + roi_shape[1] - padding_params[2] - padding_params[3] + windows_clip[3],
                cropping_params[4] + windows_clip[4],
                cropping_params[4] + roi_shape[2] - padding_params[4] - padding_params[5] + windows_clip[5],
            ),
            pred_roi=(
                padding_params[0] + windows_clip[0], roi_shape[0] - padding_params[1] + windows_clip[1],
                padding_params[2] + windows_clip[2], roi_shape[1] - padding_params[3] + windows_clip[3],
                padding_params[4] + windows_clip[4], roi_shape[2] - padding_params[5] + windows_clip[5],
            )
        )

        window_list.append((img3D_roi, gt3D_roi, box_roi, pos3D_roi))

    return window_list


if __name__ == "__main__":
    crop_transform = tio.CropOrPad(
        mask_name='label',
        target_shape=(args.crop_size, args.crop_size, args.crop_size)
    )

    infer_transform = tio.Compose([
        tio.ToCanonical(),
    ])

    test_dataset = Dataset_Union_ALL(
        paths=infer_img_data_path,
        task_names=infer_task_name,
        pathology=args.task_name,
        mode="Val",
        data_type=args.data_type,
        transform=infer_transform,
        threshold=args.threshold,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    device = args.device
    print("device:", device)

    if args.dim != 3:
        raise NotImplementedError("This script is designed for 3D inference only")

    sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    critic = Discriminator().to(device)

    if args.checkpoint_path is not None:
        model_dict = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        state_dict = remove_module_prefix(model_dict['model_state_dict'])
        sam_model_tune.load_state_dict(state_dict, strict=False)

    if args.critic_checkpoint_path is not None:
        c_model_dict = torch.load(args.critic_checkpoint_path, map_location=device, weights_only=False)
        c_state_dict = remove_module_prefix(c_model_dict['model_state_dict'])
        critic.load_state_dict(c_state_dict, strict=False)

    sam_model_tune.eval()
    critic.eval()

    sam_trans = ResizeLongestSide3D(128)
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    all_iou_list = []
    all_dice_list = []
    out_dice = dict()
    out_dice_all = OrderedDict()

    for batch_data in tqdm(test_dataloader):
        if len(batch_data) == 5:
            image3D, gt3D, boxes, texts, meta_info = batch_data
        elif len(batch_data) == 4:
            image3D, gt3D, boxes, texts = batch_data
            meta_info = None
        else:
            raise ValueError(f"Unexpected batch format length: {len(batch_data)}")

        gt3D = gt3D.type(torch.long)

        if meta_info is not None:
            img_name = meta_info[0]
            ref_seg_img = sitk.ReadImage(img_name)
            modality = osp.basename(osp.dirname(osp.dirname(osp.dirname(img_name))))
            dataset = osp.basename(osp.dirname(osp.dirname(img_name)))
            vis_root = osp.join(args.pred_output_dir, modality, dataset)
            case_base = osp.basename(img_name)
        else:
            vis_root = osp.join(args.pred_output_dir, infer_task_name[0])
            os.makedirs(vis_root, exist_ok=True)
            case_base = f"case_{len(out_dice):04d}.nii.gz"
            img_name = case_base
            ref_seg_img = None

        os.makedirs(vis_root, exist_ok=True)

        pred_path = osp.join(
            vis_root,
            case_base.replace(".nii.gz", f"_pred{args.num_clicks}.nii.gz")
        )

        iou_list, dice_list = [], []

        if args.skip_existing_pred and osp.exists(pred_path):
            pass
        else:
            image3D_full, gt3D_full, boxes_full, texts_full = image3D, gt3D, boxes, texts

            pred3D_full_dict = {
                click_idx: torch.zeros_like(gt3D_full).numpy()
                for click_idx in range(args.num_clicks + 1)
            }

            offset_mode = "center" if not args.sliding_window else "rounded"
            sliding_window_list = pad_and_crop_with_sliding_window(
                image3D_full, gt3D_full, boxes_full, crop_transform, offset_mode=offset_mode
            )

            last_points = None
            last_labels = None
            last_scribble_points = None
            last_scribble_labels = None

            for image3D_roi, gt3D_roi, boxes_roi, pos3D in sliding_window_list:
                seg_mask_list, points, labels, scribble_points, scribble_labels = finetune_model_predict3D(
                    img3D=image3D_roi,
                    gt3D=gt3D_roi,
                    boxes=boxes_roi,
                    texts=texts_full,
                    sam_model_tune=sam_model_tune,
                    critic=critic,
                    device=device,
                    click_method=args.point_method,
                    num_clicks=args.num_clicks,
                    max_scribbles=args.max_scribbles,
                    multi_click=False,
                )

                last_points = points
                last_labels = labels
                last_scribble_points = scribble_points
                last_scribble_labels = scribble_labels

                ori_roi, pred_roi = pos3D["ori_roi"], pos3D["pred_roi"]

                for idx, seg_mask in enumerate(seg_mask_list):
                    seg_mask_roi = seg_mask[
                        ...,
                        pred_roi[0]:pred_roi[1],
                        pred_roi[2]:pred_roi[3],
                        pred_roi[4]:pred_roi[5]
                    ]
                    pred3D_full_dict[idx][
                        ...,
                        ori_roi[0]:ori_roi[1],
                        ori_roi[2]:ori_roi[3],
                        ori_roi[4]:ori_roi[5]
                    ] = seg_mask_roi

            padding_params = sliding_window_list[-1][-1]["padding_params"]
            cropping_params = sliding_window_list[-1][-1]["cropping_params"]
            point_offset = np.array([
                cropping_params[0] - padding_params[0],
                cropping_params[2] - padding_params[2],
                cropping_params[4] - padding_params[4]
            ])

            points_np = [p.cpu().numpy() + point_offset for p in last_points] if last_points is not None else []
            labels_np = [l.cpu().numpy() for l in last_labels] if last_labels is not None else []
            scribble_points_np = last_scribble_points.cpu().numpy() if last_scribble_points is not None else None
            scribble_labels_np = last_scribble_labels.cpu().numpy() if last_scribble_labels is not None else None
            boxes_np = boxes_full.cpu().numpy() if boxes_full is not None else None

            prompt_info = dict(
                points=points_np,
                labels=labels_np,
                scribble_points=scribble_points_np,
                scribble_labels=scribble_labels_np,
                bbox=boxes_np,
                text=ensure_text_list(texts_full),
            )
            pt_path = osp.join(vis_root, case_base.replace(".nii.gz", "_pt.pkl"))
            pickle.dump(prompt_info, open(pt_path, "wb"))

            if args.save_image_and_gt and ref_seg_img is not None:
                save_case_array(
                    image3D_full,
                    osp.join(vis_root, case_base.replace(".nii.gz", "_img.nii.gz")),
                    ref_seg_img,
                    args.dataset_id
                )
                save_case_array(
                    gt3D_full,
                    osp.join(vis_root, case_base.replace(".nii.gz", "_gt.nii.gz")),
                    ref_seg_img,
                    args.dataset_id
                )

            if ref_seg_img is not None:
                for idx, pred3D_full in pred3D_full_dict.items():
                    save_case_array(
                        pred3D_full,
                        osp.join(vis_root, case_base.replace(".nii.gz", f"_pred{idx}.nii.gz")),
                        ref_seg_img,
                        args.dataset_id
                    )
                    pred_w_pt = overlay_points_on_pred(pred3D_full, points_np, idx, radius=2, value=10)
                    save_case_array(
                        pred_w_pt,
                        osp.join(vis_root, case_base.replace(".nii.gz", f"_pred{idx}_wPt.nii.gz")),
                        ref_seg_img,
                        args.dataset_id
                    )

        if ref_seg_img is not None:
            if args.dataset_id in [0, 1]:
                for click_idx in range(args.num_clicks + 1):
                    reorient_tensor = lambda in_arr: np.transpose(in_arr.squeeze().detach().cpu().numpy(), (2, 1, 0))
                    curr_pred_path = osp.join(vis_root, case_base.replace(".nii.gz", f"_pred{click_idx}.nii.gz"))
                    medsam_seg = sitk.GetArrayFromImage(sitk.ReadImage(curr_pred_path))
                    iou_list.append(round(compute_iou(medsam_seg, reorient_tensor(gt3D_full)), 4))
                    dice_list.append(round(compute_dice(reorient_tensor(gt3D_full), medsam_seg), 4))
            elif args.dataset_id == 13:
                for click_idx in range(args.num_clicks + 1):
                    reorient_tensor = lambda in_arr: in_arr.squeeze().detach().cpu().numpy()
                    curr_pred_path = osp.join(vis_root, case_base.replace(".nii.gz", f"_pred{click_idx}.nii.gz"))
                    medsam_seg = sitk.GetArrayFromImage(sitk.ReadImage(curr_pred_path))
                    iou_list.append(round(compute_iou(medsam_seg, reorient_tensor(gt3D_full)), 4))
                    dice_list.append(round(compute_dice(reorient_tensor(gt3D_full), medsam_seg), 4))
            else:
                for click_idx in range(args.num_clicks + 1):
                    reorient_tensor = lambda in_arr: np.transpose(in_arr.squeeze().detach().cpu().numpy(), (2, 1, 0))
                    curr_pred_path = osp.join(vis_root, case_base.replace(".nii.gz", f"_pred{click_idx}.nii.gz"))
                    medsam_seg = sitk.GetArrayFromImage(sitk.ReadImage(curr_pred_path))
                    iou_list.append(round(compute_iou(medsam_seg, reorient_tensor(gt3D_full)), 4))
                    dice_list.append(round(compute_dice(reorient_tensor(gt3D_full), medsam_seg), 4))

            del reorient_tensor, medsam_seg
            gc.collect()
            torch.cuda.empty_cache()

            per_iou = np.nanmax(iou_list)
            per_dice = np.nanmax(dice_list)
            all_iou_list.append(per_iou)
            all_dice_list.append(per_dice)
            out_dice[img_name] = per_dice

            cur_dice_dict = OrderedDict()
            for i, dice in enumerate(dice_list):
                cur_dice_dict[str(i)] = dice
            out_dice_all[img_name] = cur_dice_dict

            print(dice_list)

        del image3D, gt3D
        gc.collect()
        torch.cuda.empty_cache()

    mean_iou = np.nanmean(all_iou_list) if len(all_iou_list) > 0 else np.nan
    mean_dice = np.nanmean(all_dice_list) if len(all_dice_list) > 0 else np.nan

    with open(output_file, 'w') as f:
        f.write(f"File save name: {args.file_save_name}\n")
        f.write(f"Mean IoU: {mean_iou}\n")
        f.write(f"Mean Dice: {mean_dice}\n")

    print(f"Results saved to {output_file}")

    final_dice_dict = OrderedDict()
    for k, _ in out_dice_all.items():
        organ = k.split('/')[-4] if '/' in k else infer_task_name[0]
        final_dice_dict[organ] = OrderedDict()

    for k, v in out_dice_all.items():
        organ = k.split('/')[-4] if '/' in k else infer_task_name[0]
        final_dice_dict[organ][k] = v

    if args.split_num > 1:
        args.save_name = args.save_name.replace('.py', f'_s{args.split_num}i{args.split_idx}.py')

    print("Save to", args.save_name)
    with open(args.save_name, 'w') as f:
        f.writelines(f'# mean dice: \t{mean_dice}\n')
        f.writelines('dice_Ts = {')
        for k, v in out_dice.items():
            f.writelines(f'\'{str(k)}\': {v},\n')
        f.writelines('}')

    with open(args.save_name.replace('.py', '.json'), 'w') as f:
        json.dump(final_dice_dict, f, indent=4)

    print("Done")