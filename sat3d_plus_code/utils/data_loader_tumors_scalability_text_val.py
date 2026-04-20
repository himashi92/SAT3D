import os
import torch
import SimpleITK as sitk
import torchio as tio
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator
from utils.tumor_data_paths_full_dataset_scalability_text import class_mapping, all_datasets, img_datas

def get_label_index(dataset_name, label):
    labels = class_mapping[dataset_name]['labels']
    if label in labels:
        return labels.index(label)  # Returns the position in the list
    else:
        raise ValueError(f"Label '{label}' not found in dataset '{dataset_name}'")

class Dataset_Union_ALL(Dataset):
    def __init__(self, paths, task_names, pathology, mode='train', data_type='Tr', image_size=128,
                 transform=None, threshold=500, split_num=1, split_idx=0, pcc=False):
        self.paths = paths
        self.task_names = task_names
        self.data_type = data_type
        self.split_num = split_num
        self.split_idx = split_idx
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        self.pcc = pcc

        self.image_paths = []
        self.label_paths = []
        self.classes_per_sample = []          # class label (e.g., 1,2,4)
        self.modality_texts_per_sample = []    # store the text description

        self.class_idx = get_label_index(self.task_names[0], pathology)

        # Iterate over tasks and corresponding paths
        for task_name, path in zip(self.task_names, self.paths):
            self.classes = [class_mapping[task_name]['class'][self.class_idx]]
            self.labels = [class_mapping[task_name]['labels'][self.class_idx]]
            self.modality_texts = [class_mapping[task_name]['modality'][self.class_idx]]   # list of strings
            self._set_file_paths(path, task_name)

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        # Load the image and label
        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        # Adjust image metadata to match the label
        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        # Convert image and label to torchio format
        subject = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
        )

        # Apply clamping for CT images
        if '_ct/' in self.image_paths[index]:
            if any(x in self.image_paths[index] for x in ['Liver', 'Lung', 'HepaticVessel', 'Pancreas', 'Colon', 'KiPA22', 'KiTS23']):
                subject = tio.Clamp(-1000, 1000)(subject)
            else:
                subject = tio.Clamp(-1000, 1000)(subject)

        # Apply transformations, if any
        if self.transform:
            try:
                subject = self.transform(subject)
            except Exception as e:
                print(f"Error during transformation: {e}")
                print(self.image_paths[index])

        # Get the binary segmentation mask for the specified class
        cls = self.classes_per_sample[index]
        binary_label = (subject.label.data == cls).float()  # (1, D, H, W)

        # Skip if no segmentation exists
        if torch.sum(binary_label) == 0:
            return self.__getitem__((index + 1) % len(self))

        # ---- Compute bounding box from the binary mask ----
        fg_coords = torch.nonzero(binary_label[0] > 0).float()  # (N, 3) in (z, y, x) order
        if fg_coords.numel() > 0:
            # Reorder to (x, y, z) as expected by the prompt encoder
            fg_coords = fg_coords[:, [2, 1, 0]]  # now (x, y, z)
            x_min, y_min, z_min = fg_coords.min(dim=0)[0]
            x_max, y_max, z_max = fg_coords.max(dim=0)[0]
            bbox = torch.tensor([x_min, y_min, z_min, x_max, y_max, z_max])  # (6,)
        else:
            bbox = torch.zeros(6)  # fallback (should not happen)

        # Get the modality text for this sample
        modality_text = self.modality_texts_per_sample[index]

        if self.mode == "train" and self.data_type == 'Tr':
            return (
                subject.image.data.clone().detach().float(),
                binary_label.clone().detach().float(),
                bbox,
                modality_text          # string
            )
        else:
            return (
                subject.image.data.clone().detach().float(),
                binary_label.clone().detach().float(),
                bbox,
                modality_text,
                self.image_paths[index]
            )

    def _set_file_paths(self, path, task_name):
        d = os.path.join(path, f'labels{self.data_type}')
        if os.path.exists(d):
            for name in os.listdir(d):
                base = os.path.basename(name).split('.nii.gz')[0]

                # Generate paths based on task-specific conventions (as in your original)
                if task_name == 'HECKTOR22_ct':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{base}__CT.nii.gz')
                elif task_name == 'HECKTOR22_pet':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{base}__PT.nii.gz')
                elif task_name == 'HNTSMRG24_mr_t2':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = base.replace('mask', 'T2')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                elif task_name == 'BraTS_2021_mr_t1ce':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = base.replace('_seg', '_t1ce')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                elif task_name == 'BraTS_2021_mr_flair':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = base.replace('_seg', '_flair')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                elif task_name == 'BraTS_2021_mr_t1':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = base.replace('_seg', '_t1')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                elif task_name == 'BraTS_2021_mr_t2':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = base.replace('_seg', '_t2')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                elif task_name == 'Autopet_ct':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = str(base + '_0000')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                elif task_name == 'Autopet_pet':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = str(base + '_0001')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                elif task_name == 'TDSC_ABUS':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = base.replace('MASK', 'DATA')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                else:
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    img_path = label_path.replace('labels', 'images')

                # Add each class label as a separate entry
                for i, cls in enumerate(self.classes):
                    self.image_paths.append(img_path)
                    self.label_paths.append(label_path)
                    self.classes_per_sample.append(cls)
                    # Store the corresponding text description for this class
                    self.modality_texts_per_sample.append(self.modality_texts[i])


class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())