# set up environment
import datetime
import logging
import random

import matplotlib
import numpy as np
from tqdm import tqdm
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import gc
join = os.path.join
from torch.backends import cudnn
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
from segment_anything_with_swin_conf_plus.build_samswin3D import sam_model_registry3D
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader_tumors_scalability_text import Dataset_Union_ALL, Union_Dataloader
from utils.tumor_data_paths_full_dataset_scalability_text import img_datas, all_datasets
from networks import Discriminator
from monai.losses import DiceCELoss

import warnings
warnings.filterwarnings("ignore")

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='sat3D_4')
parser.add_argument('--resume_task_name', type=str, default='sat3D_4_plus')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='swin2')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='./work_dir')
parser.add_argument('--resume_work_dir', type=str, default='./work_dir')

# train
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--resume', action='store_true', default=True)
parser.add_argument('--allow_partial_weight', action='store_true', default=True)

# lr_scheduler
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--accumulation_steps', type=int, default=20)

#CHANGED
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--warmup_epoch', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=3)

parser.add_argument('--dist', dest='dist', type=bool, default=True,
                    help='distributed training or not')
parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
parser.add_argument('--init_method', type=str, default="env://")
parser.add_argument('--bucket_cap_mb', type=int, default=25,
                    help='The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)')

args = parser.parse_args()


device = args.device
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def build_model(args, gpu):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(args.gpu)
    critic_model = Discriminator().to(args.gpu)

    sam_model = torch.nn.parallel.DistributedDataParallel(
        sam_model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb
    )

    critic_model = torch.nn.parallel.DistributedDataParallel(
        critic_model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb
    )
    return sam_model, critic_model


def get_dataloaders(args):
    train_dataset = Dataset_Union_ALL(
        paths=img_datas,
        task_names=all_datasets,
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(mask_name='label', target_shape=(args.img_size, args.img_size, args.img_size)),
            tio.RandomFlip(axes=(0, 1, 2)),
        ]),
        threshold=1000
    )

    train_sampler = DistributedSampler(train_dataset)

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataloader


class BaseTrainer:
    def __init__(self, model, critic, dataloaders, args, gpu):

        self.model = model
        self.gpu = gpu
        self.critic = critic
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()

        if (args.resume):
            self.init_checkpoint(join(self.args.resume_work_dir, self.args.resume_task_name, 'sam_model_latest.pth'),
                                 join(self.args.resume_work_dir, self.args.resume_task_name, 'critic_latest.pth'))
        else:
            self.start_epoch = 0

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def set_optimizer(self):
        sam_model = self.model.module
        critic = self.critic.module

        self.optimizer = torch.optim.AdamW([
            {'params': sam_model.image_encoder.parameters()},
            {'params': sam_model.prompt_encoder.parameters(), 'lr': self.args.lr * 0.1},
            {'params': sam_model.mask_decoder.parameters(), 'lr': self.args.lr * 0.1},
        ], lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)

        self.dis_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr)

    def set_lr_scheduler(self):
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)
        self.c_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.dis_optimizer, T_max=args.num_epochs)

    def init_checkpoint(self, ckp_path, critic_ckp_path):
        last_ckpt = None
        critic_last_ckpt = None

        loc = 'cuda:{}'.format(self.gpu)
        if os.path.exists(ckp_path):
            dist.barrier()
            print("pretrained checkpoints exists!")
            last_ckpt = torch.load(ckp_path, map_location=loc, weights_only=False)
            critic_last_ckpt = torch.load(critic_ckp_path, map_location=loc, weights_only=False)

        if last_ckpt:
            model_state = last_ckpt['model_state_dict']
            critic_state = critic_last_ckpt['model_state_dict']

            # Handle 'module.' prefix mismatches
            if list(model_state.keys())[0].startswith('module.') and not hasattr(self.model, 'module'):
                model_state = {k[7:]: v for k, v in model_state.items()}
            elif not list(model_state.keys())[0].startswith('module.') and hasattr(self.model, 'module'):
                model_state = {'module.' + k: v for k, v in model_state.items()}

            if list(critic_state.keys())[0].startswith('module.') and not hasattr(self.critic, 'module'):
                critic_state = {k[7:]: v for k, v in critic_state.items()}
            elif not list(critic_state.keys())[0].startswith('module.') and hasattr(self.critic, 'module'):
                critic_state = {'module.' + k: v for k, v in critic_state.items()}

            if self.args.allow_partial_weight:
                self.model.load_state_dict(model_state, strict=False)
                self.critic.load_state_dict(critic_state, strict=False)
                print("Loaded checkpoint with strict=False (partial weights allowed)")
            else:
                self.model.load_state_dict(model_state)
                self.critic.load_state_dict(critic_state)

            if not self.args.resume:
                self.start_epoch = 0
            else:
                self.start_epoch = last_ckpt['epoch']
                try:
                    self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                except ValueError as e:
                    print(f"Warning: Could not load optimizer state: {e}. Optimizer will be reinitialized.")
                if 'lr_scheduler_state_dict' in last_ckpt:
                    try:
                        self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                    except Exception as e:
                        print(f"Warning: Could not load scheduler state: {e}. Scheduler will restart.")
                self.losses = last_ckpt.get('losses', [])
                self.dices = last_ckpt.get('dices', [])
                self.best_loss = last_ckpt.get('best_loss', np.inf)
                self.best_dice = last_ckpt.get('best_dice', 0.0)

            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "args": self.args,
            "used_datas": img_datas,
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))

    def save_critic_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.dis_optimizer.state_dict(),
            "lr_scheduler_state_dict": self.c_lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "args": self.args,
            "used_datas": img_datas,
        }, join(MODEL_SAVE_PATH, f"critic_{describe}.pth"))

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, low_res_conf, points=None, boxes=None, text=None):
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
            text=text,          # list of strings
        )

        # Run decoder WITHOUT mixed precision to avoid CUBLAS errors
        with amp.autocast(enabled=False):
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding.float().to(device),
                image_pe=sam_model.prompt_encoder.get_dense_pe().float(),
                sparse_prompt_embeddings=sparse_embeddings.float(),
                dense_prompt_embeddings=dense_embeddings.float(),
                multimask_output=False,
            )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        points, labels = click_methods[self.args.click_type](prev_masks, gt3D)

        # ----- Debug: validate point coordinates -----
        if isinstance(points, list):
            for p in points:
                assert p.min() >= 0 and p.max() < args.img_size, f"Click coordinates out of bounds: {p}"
        else:
            assert points.min() >= 0 and points.max() < args.img_size, f"Click coordinates out of bounds: {points}"
        # ---------------------------------------------

        # Convert list to batched tensor if needed
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

        points = points.to(self.args.gpu)
        labels = labels.to(self.args.gpu)

        self.click_points.append(points)
        self.click_labels.append(labels)

        points_multi = torch.cat(self.click_points, dim=1)   # (B, total_pts, 3)
        labels_multi = torch.cat(self.click_labels, dim=1)   # (B, total_pts)

        if self.args.multi_click:
            return points_multi, labels_multi
        else:
            return points, labels

    def sample_positive_scribbles(self, gt3D, max_scribbles=10):
        """
        Randomly sample positive scribble points from GT foreground only.

        Args:
            gt3D: tensor of shape (B, 1, D, H, W)
            max_scribbles: maximum number of positive scribble points per sample

        Returns:
            scribble_points: (B, N, 3) in (x, y, z) order
            scribble_labels: (B, N) all ones
        """
        batch_points = []
        batch_labels = []

        B = gt3D.shape[0]
        device = gt3D.device

        for b in range(B):
            fg_coords = torch.nonzero(gt3D[b, 0] > 0, as_tuple=False).float()  # (N, 3) in (z, y, x)

            if fg_coords.numel() == 0:
                pts = torch.zeros((0, 3), device=device, dtype=torch.float32)
                lbs = torch.zeros((0,), device=device, dtype=torch.long)
            else:
                num_pts = min(max_scribbles, fg_coords.shape[0])

                # randomly sample only from positive tumour region
                rand_idx = torch.randperm(fg_coords.shape[0], device=device)[:num_pts]
                pts = fg_coords[rand_idx][:, [2, 1, 0]]  # convert (z, y, x) -> (x, y, z)
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

            scribble_points = torch.stack(padded_points, dim=0)   # (B, N, 3)
            scribble_labels = torch.stack(padded_labels, dim=0)   # (B, N)

        return scribble_points, scribble_labels

    def interaction(self, sam_model, critic, image_embedding, gt3D, num_clicks, box, text):
        # Sample random positive scribbles directly from GT each iteration
        scribble_points, scribble_labels = self.sample_positive_scribbles(
            gt3D, max_scribbles=10
        )

        if scribble_points.numel() > 0:
            scribble_points = scribble_points.to(self.args.gpu)
            scribble_labels = scribble_labels.to(self.args.gpu)

        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(self.args.gpu)
        low_res_masks = F.interpolate(
            prev_masks.float(),
            size=(args.img_size // 4, args.img_size // 4, args.img_size // 4)
        )

        self.click_points = []
        self.click_labels = []

        # Add scribbles as initial positive prompts
        if scribble_points.numel() > 0:
            self.click_points.append(scribble_points)
            self.click_labels.append(scribble_labels)

        if box is not None:
            box = box.view(-1, 2, 3).to(self.args.gpu)
        else:
            box = None

        initial_points = (scribble_points, scribble_labels) if scribble_points.numel() > 0 else None

        zero_conf = torch.zeros_like(low_res_masks).to(self.args.gpu)

        # First forward with initial prompts
        low_res_masks, prev_masks = self.batch_forward(
            sam_model,
            image_embedding,
            gt3D,
            low_res_masks,
            low_res_conf=zero_conf,
            points=initial_points,
            boxes=box,
            text=text
        )
        loss = self.seg_loss(prev_masks, gt3D)
        return_loss += loss

        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)

            conf_map = (torch.sigmoid(critic(torch.sigmoid(prev_masks).float())).to(self.args.gpu) > 0.5).float()
            low_res_conf = F.interpolate(
                conf_map.float(),
                size=(args.img_size // 4, args.img_size // 4, args.img_size // 4)
            )

            low_res_masks, prev_masks = self.batch_forward(
                sam_model,
                image_embedding,
                gt3D,
                low_res_masks,
                low_res_conf,
                points=(points_input, labels_input),
                boxes=box,
                text=text
            )

            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss

        return prev_masks, return_loss

    def get_dice_score(self, prev_masks, gt3D):
        smooth = 0.0001
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5
            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return 0.0
            volume_intersect = (mask_gt & mask_pred).sum()
            return (2 * volume_intersect + smooth) / (volume_sum + smooth)
        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()

    def disc_loss(self, pred, target):
        CE = torch.nn.BCEWithLogitsLoss()
        real_loss1 = CE(target, torch.ones_like(target).float())
        fake_loss1 = CE(pred, torch.zeros_like(pred).float())
        loss = (1 / 2) * (real_loss1 + fake_loss1)
        return loss

    def gen_loss(self, pred):
        CE = torch.nn.BCEWithLogitsLoss()
        fake_loss1 = CE(pred, torch.ones_like(pred).float())
        return fake_loss1

    def loss_mask(self, u_prediction_1, label, critic_segs, T_m=0.3):
        CE = torch.nn.BCEWithLogitsLoss()
        gen_mask = (critic_segs.squeeze(0) > T_m).float()
        label = label.float()
        loss_a = gen_mask * CE(u_prediction_1, label)
        return loss_a.mean()

    def train_epoch(self, epoch, args, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        epoch_dice = 0

        self.model.train()
        self.critic.train()

        sam_model = self.model.module
        critic = self.critic.module

        epoch_iterator = tqdm(
            self.dataloaders, desc=f"[RANK {args.rank}: GPU {args.gpu}]", dynamic_ncols=True
        )

        self.optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        step_loss = 0

        for step, batch in enumerate(epoch_iterator):
            if len(batch) == 7:  # includes paths
                image3D, gt3D, boxes, texts, _ = batch
            else:
                image3D, gt3D, boxes, texts = batch

            image3D = self.norm_transform(image3D.squeeze(dim=1))
            image3D = image3D.unsqueeze(dim=1)
            image3D = image3D.to(self.args.gpu)
            gt3D = gt3D.to(self.args.gpu).type(torch.long)


            with amp.autocast():
                image_embedding = sam_model.image_encoder(image3D)

                self.click_points = []
                self.click_labels = []

                pred_list = []

                prev_masks, loss_sam = self.interaction(
                    sam_model, critic, image_embedding, gt3D, num_clicks=11,
                    box=boxes, text=texts   # list of strings
                )
                prev_masks_sig = torch.sigmoid(prev_masks)
                g_critic_segs_1 = critic(prev_masks_sig)
                g_critic_segs_2 = critic(gt3D.float())

                loss_adversarial_gen = self.gen_loss(g_critic_segs_1)
                loss_adversarial_critic = self.disc_loss(g_critic_segs_1, g_critic_segs_2)

                dsc = self.get_dice_score(prev_masks_sig, gt3D)

                critic_segs = torch.sigmoid(g_critic_segs_1)
                loss_uncertainty = self.loss_mask(prev_masks, gt3D, critic_segs)

                loss = loss_sam + 0.01 * loss_adversarial_gen + 0.1 * loss_uncertainty

            epoch_loss += loss.item()
            epoch_dice += dsc
            cur_loss = loss.item()

            loss /= self.args.accumulation_steps
            self.scaler.scale(loss).backward()

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(torch.sigmoid(prev_masks), gt3D)

                loss_adversarial_critic = loss_adversarial_critic.clone().detach().requires_grad_(True)

                self.scaler.scale(loss_adversarial_critic).backward()
                self.scaler.step(self.dis_optimizer)
                self.scaler.update()
                self.dis_optimizer.zero_grad()
            else:
                step_loss += cur_loss

            if step % self.args.accumulation_steps == 0 and step != 0:
                print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}')
                if print_dice > self.step_best_dice:
                    self.step_best_dice = print_dice
                    if print_dice > 0.9:
                        self.save_checkpoint(
                            epoch,
                            sam_model.state_dict(),
                            describe=f'{epoch}_step_dice:{print_dice}_best'
                        )
                if print_loss < self.step_best_loss:
                    self.step_best_loss = print_loss

        epoch_loss /= step + 1
        epoch_dice /= step + 1

        del g_critic_segs_1, g_critic_segs_2, loss_adversarial_critic, loss_adversarial_gen, loss_sam, image3D, gt3D
        gc.collect()
        torch.cuda.empty_cache()

        return epoch_loss, epoch_iou, epoch_dice, pred_list

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()

    def train(self, args):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            dist.barrier()
            self.dataloaders.sampler.set_epoch(epoch)

            num_clicks = np.random.randint(1, 11)
            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(epoch, args, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.c_lr_scheduler is not None:
                self.c_lr_scheduler.step()

            dist.barrier()

            if self.args.rank == 0:
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}')

                state_dict = self.model.state_dict()
                state_dict_critic = self.critic.state_dict()

                self.save_checkpoint(epoch, state_dict, describe='latest')
                self.save_critic_checkpoint(epoch, state_dict_critic, describe='latest')

                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save_checkpoint(epoch, state_dict, describe='loss_best')
                    self.save_critic_checkpoint(epoch, state_dict_critic, describe='loss_best')
                if epoch_dice > self.best_dice:
                    self.best_dice = epoch_dice
                    self.save_checkpoint(epoch, state_dict, describe='dice_best')
                    self.save_critic_checkpoint(epoch, state_dict_critic, describe='dice_best')

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')

        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')


def main():
    torch.manual_seed(2025)
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12222'

    ngpus_per_node = torch.cuda.device_count()
    print("Spwaning processces, ngpus_per_node={}".format(ngpus_per_node))
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = ngpus_per_node
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    args.gpu = gpu
    args.rank = rank
    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.init_method,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=36000)
    )
    print('init_process_group finished')

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))

    dataloaders = get_dataloaders(args)
    model, critic = build_model(args, gpu)
    trainer = BaseTrainer(model, critic, dataloaders, args, gpu)
    trainer.train(args)
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()