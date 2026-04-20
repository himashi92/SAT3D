import os
import torch
import numpy as np
import SimpleITK as sitk
import torchio as tio
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator
from utils.tumor_data_paths import class_mapping, all_datasets, img_datas


class Dataset_Union_ALL(Dataset):
    def __init__(self, paths, task_names, mode='train', data_type='Tr', image_size=128,
                 transform=None, threshold=500, split_num=1, split_idx=0, pcc=False):
        self.paths = paths
        self.task_names = task_names  # Handle multiple task names
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

        # Iterate over tasks and corresponding paths
        for task_name, path in zip(self.task_names, self.paths):
            # Get the class mapping for each specific task
            self.classes = class_mapping[task_name]['class']
            self.labels = class_mapping[task_name]['labels']
            self._set_file_paths(path, task_name)

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        subject = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
        )

        if '_ct/' in self.image_paths[index]:
            subject = tio.Clamp(-1000, 1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])

        # Generate binary segmentation maps for each class
        binary_segmentations = []
        segmentation_exists = False

        for cls in self.classes:
            binary_label = (subject.label.data == cls).float()
            if torch.sum(binary_label) > 0:
                segmentation_exists = True
            binary_segmentations.append(binary_label)

        # Skip sample if no segmentation exists for the specified class
        if not segmentation_exists:
            return self.__getitem__((index + 1) % len(self))  # Skip and load the next sample

        if self.mode == "train" and self.data_type == 'Tr':
            return subject.image.data.clone().detach(), binary_segmentations
        else:
            return subject.image.data.clone().detach(), binary_segmentations, self.image_paths[index]

    def _set_file_paths(self, path, task_name):
        # if ${path}/labelsTr exists, search all .nii.gz
        d = os.path.join(path, f'labels{self.data_type}')
        if os.path.exists(d):
            for name in os.listdir(d):
                base = os.path.basename(name).split('.nii.gz')[0]

                # Handle various task-specific cases
                if task_name == 'HECKTOR22_ct':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    img_path = os.path.join(path, f'labels{self.data_type}', f'{base}__CT.nii.gz')
                    self.image_paths.append(img_path.replace('labels', 'images'))
                    self.label_paths.append(label_path)
                elif task_name == 'HECKTOR22_pet':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    img_path = os.path.join(path, f'labels{self.data_type}', f'{base}__PT.nii.gz')
                    self.image_paths.append(img_path.replace('labels', 'images'))
                    self.label_paths.append(label_path)
                elif task_name == 'HNTSMRG24_mr_t2':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = base.replace('mask', 'T2')
                    img_path = os.path.join(path, f'labels{self.data_type}', f'{rbase}.nii.gz')
                    self.image_paths.append(img_path.replace('labels', 'images'))
                    self.label_paths.append(label_path)
                elif task_name.startswith('BraTS_2021_mr'):
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = base.replace('_seg', task_name.split('BraTS_2021_mr')[1])
                    img_path = os.path.join(path, f'labels{self.data_type}', f'{rbase}.nii.gz')
                    self.image_paths.append(img_path.replace('labels', 'images'))
                    self.label_paths.append(label_path)
                else:
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    self.image_paths.append(label_path.replace('labels', 'images'))
                    self.label_paths.append(label_path)



class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == "__main__":
    # Define multiple tasks and corresponding paths
    tasks = all_datasets
    paths = img_datas

    # Create the dataset
    test_dataset = Dataset_Union_ALL(
        paths=paths,
        task_names=tasks,
        data_type='Ts',
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(mask_name='label', target_shape=(128, 128, 128)),
        ]),
        threshold=0
    )

    # Create the dataloader
    test_dataloader = Union_Dataloader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1,
        shuffle=True
    )

    # Loop over the data loader to generate binary segmentation for each class
    for batch in test_dataloader:
        if batch is None:
            continue  # Skip samples where there is no segmentation for all classes
        image, binary_segmentations, n = batch
        print(n)
        print(f"Image Shape: {image.shape}")
        for idx, binary_segmentation in enumerate(binary_segmentations):
            print(f"Binary Segmentation Shape for class {idx + 1}: {binary_segmentation.shape}")
            print(f"Unique values in the segmentation for class {idx + 1}: {torch.unique(binary_segmentation)}")
        continue



