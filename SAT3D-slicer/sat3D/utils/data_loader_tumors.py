import os
import torch
import SimpleITK as sitk
import torchio as tio
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator
from utils.tumor_data_paths_full_dataset_v2 import class_mapping, all_datasets, img_datas


class Dataset_Union_ALL(Dataset):
    def __init__(self, paths, task_names, mode='train', data_type='Tr', image_size=128,
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
        self.classes_per_sample = []  # Stores the class for each label path entry

        # Iterate over tasks and corresponding paths
        for task_name, path in zip(self.task_names, self.paths):
            self.classes = class_mapping[task_name]['class']
            self.labels = class_mapping[task_name]['labels']
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
            if 'Liver' in self.image_paths[index] or 'Lung' in self.image_paths[index] or 'HepaticVessel' in self.image_paths[index] or 'Pancreas' in self.image_paths[index] or 'Colon' in self.image_paths[index] or 'KiPA22' in self.image_paths[index] or 'KiTS23' in self.image_paths[index]:
                subject = tio.Clamp(-1000, 1000)(subject)
            else:
                subject = tio.Clamp(-1000, 1000)(subject)
                # resampler = tio.Resample(target=(1.5, 1.01821005, 1.01821005), image_interpolation="linear")
                # subject = resampler(subject)

        # if '_pet/' in self.image_paths[index]:
        #     #subject = tio.ToCanonical()(subject)
        #     resampler = tio.Resample(target=(1.5, 1.01821005, 1.01821005), image_interpolation="linear")
        #     subject = resampler(subject)

        # Apply transformations, if any
        if self.transform:
            try:
                subject = self.transform(subject)
            except Exception as e:
                print(f"Error during transformation: {e}")
                print(self.image_paths[index])

        # Get the binary segmentation mask for the specified class
        cls = self.classes_per_sample[index]
        binary_label = (subject.label.data == cls).float()
        if torch.sum(binary_label) == 0:  # Skip if no segmentation exists
            return self.__getitem__((index + 1) % len(self))

        if self.mode == "train" and self.data_type == 'Tr':
            return subject.image.data.clone().detach().long(), binary_label.clone().detach().float()
        else:
            return subject.image.data.clone().detach().long(), binary_label.clone().detach().float() , self.image_paths[index]

    def _set_file_paths(self, path, task_name):
        # Locate all label files
        d = os.path.join(path, f'labels{self.data_type}')
        if os.path.exists(d):
            for name in os.listdir(d):
                base = os.path.basename(name).split('.nii.gz')[0]

                # Generate paths based on task-specific conventions
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
                elif task_name=='BraTS_2021_mr_t1ce':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = base.replace('_seg', '_t1ce')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                elif task_name=='BraTS_2021_mr_flair':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = base.replace('_seg', '_flair')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                elif task_name=='BraTS_2021_mr_t1':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = base.replace('_seg', '_t1')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                elif task_name=='BraTS_2021_mr_t2':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = base.replace('_seg', '_t2')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                elif task_name=='Autopet_ct':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = str(base+'_0000')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                elif task_name=='Autopet_pet':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = str(base+'_0001')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                elif task_name=='TDSC_ABUS':
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    rbase = base.replace('MASK', 'DATA')
                    img_path = os.path.join(path, f'images{self.data_type}', f'{rbase}.nii.gz')
                else:
                    label_path = os.path.join(path, f'labels{self.data_type}', f'{base}.nii.gz')
                    img_path = label_path.replace('labels', 'images')

                # Add each class label as a separate entry
                for cls in self.classes:
                    self.image_paths.append(img_path)
                    self.label_paths.append(label_path)
                    self.classes_per_sample.append(cls)


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
        image, binary_segmentation, n = batch
        print(n)
        print(f"Image Shape: {image.shape}")
        print(f"Binary Segmentation Shape: {binary_segmentation.shape}")
        print(f"Unique values in the segmentation: {torch.unique(binary_segmentation)}")



