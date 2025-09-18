# set up environment
import datetime
import logging
import matplotlib
import numpy as np
import random
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
from segment_anything_with_swin_conf.build_samswin3D import sam_model_registry3D
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from utils.click_method import get_next_click3D_torch_2, get_next_click3D_torch_with_dice_rev
from utils.data_loader_tumors import Dataset_Union_ALL, Union_Dataloader
from utils.tumor_data_paths import img_datas, all_datasets
from networks import Discriminator
from monai.losses import DiceCELoss

import warnings

warnings.filterwarnings("ignore")

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='sat3D')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='swin_c')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='./work_dir')

# train
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--resume', action='store_true', default=True)
parser.add_argument('--allow_partial_weight', action='store_true', default=False)

# lr_scheduler
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--accumulation_steps', type=int, default=20)

# CHANGED
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--warmup_epoch', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=500)
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
    train_dataset = Dataset_Union_ALL(paths=img_datas, task_names=all_datasets, transform=tio.Compose([
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.img_size, args.img_size, args.img_size)),
        # crop only object region
        tio.RandomFlip(axes=(0, 1, 2)),
    ]),
                                      threshold=1000)

    train_sampler = DistributedSampler(train_dataset)

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
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
            self.init_checkpoint(join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'),
                                 join(self.args.work_dir, self.args.task_name, 'critic_latest.pth'))
        else:
            self.start_epoch = 0

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def set_optimizer(self):
        sam_model = self.model.module
        critic = self.critic.module

        self.optimizer = torch.optim.AdamW([
            {'params': sam_model.image_encoder.parameters()},  # , 'lr': self.args.lr * 0.1},
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
            last_ckpt = torch.load(ckp_path, map_location=loc)
            critic_last_ckpt = torch.load(critic_ckp_path, map_location=loc)

        if last_ckpt:
            self.model.load_state_dict(last_ckpt['model_state_dict'])
            self.critic.load_state_dict(critic_last_ckpt['model_state_dict'])

            if not self.args.resume:
                self.start_epoch = 0
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                # self.dis_optimizer.load_state_dict(critic_last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
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

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, low_res_conf, points=None):

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
            conf=low_res_conf
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(self.args.gpu),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(self.args.gpu)
        points_la = torch.cat(batch_labels, dim=0).to(self.args.gpu)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(self.args.gpu)
        labels_multi = torch.cat(self.click_labels, dim=1).to(self.args.gpu)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

    def interaction(self, sam_model, critic, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(self.args.gpu)
        random_insert = np.random.randint(2, 9)

        low_res_masks = F.interpolate(prev_masks.float(),
                                      size=(args.img_size // 4, args.img_size // 4, args.img_size // 4))

        for num_click in range(num_clicks):
            conf_map = (torch.sigmoid(critic(torch.sigmoid(prev_masks).float())).to(self.args.gpu) > 0.5).float()

            low_res_conf = F.interpolate(conf_map.float(),
                                         size=(args.img_size // 4, args.img_size // 4, args.img_size // 4))

            points_input, labels_input = self.get_points(prev_masks, gt3D)

            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks,
                                                               low_res_conf,
                                                               points=None)
            else:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks,
                                                               low_res_conf,
                                                               points=[points_input, labels_input])

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

        loss = fake_loss1

        return loss

    def loss_mask(self, u_prediction_1, label, critic_segs, T_m=0.3):
        CE = torch.nn.BCEWithLogitsLoss()
        gen_mask = (critic_segs.squeeze(0) > T_m).float()
        label = label.float()
        loss_a = gen_mask * CE(u_prediction_1, label)

        loss_diff_avg = loss_a.mean()

        return loss_diff_avg

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

        for step, (image3D, gt3D) in enumerate(epoch_iterator):
            image3D = self.norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
            image3D = image3D.unsqueeze(dim=1)

            image3D = image3D.to(self.args.gpu)

            gt3D = gt3D.to(self.args.gpu).type(torch.long)
            with amp.autocast():
                image_embedding = sam_model.image_encoder(image3D)

                self.click_points = []
                self.click_labels = []

                pred_list = []

                prev_masks, loss_sam = self.interaction(sam_model, critic, image_embedding, gt3D, num_clicks=11)
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

            if (self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}')

                state_dict = self.model.state_dict()
                state_dict_critic = self.critic.state_dict()

                # save latest checkpoint
                self.save_checkpoint(
                    epoch,
                    state_dict,
                    describe='latest'
                )

                self.save_critic_checkpoint(
                    epoch,
                    state_dict_critic,
                    describe='latest'
                )

                # save train loss best checkpoint
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='loss_best'
                    )

                    self.save_critic_checkpoint(
                        epoch,
                        state_dict_critic,
                        describe='loss_best'
                    )

                # save train dice best checkpoint
                if epoch_dice > self.best_dice:
                    self.best_dice = epoch_dice
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='dice_best'
                    )

                    self.save_critic_checkpoint(
                        epoch,
                        state_dict_critic,
                        describe='dice_best'
                    )

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')


def main():
    # set seeds
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
    world_size = ngpus_per_node  # args.world_size
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
