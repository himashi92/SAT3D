img_datas = [
    '/home/data/tumor3D_data/Autopet_ct/split/',
    '/home/data/tumor3D_data/Autopet_pet/split/',
    '/home/data/tumor3D_data/BraTS_2021_mr_t1/split/',
    '/home/data/tumor3D_data/BraTS_2021_mr_t2/split/',
    '/home/data/tumor3D_data/BraTS_2021_mr_flair/split/',
    '/home/data/tumor3D_data/BraTS_2021_mr_t1ce/split/',
    '/home/data/tumor3D_data/HNTSMRG24_mr_t2/split/',
    '/home/data/tumor3D_data/Task06_Lung_ct/split/',
    '/home/data/tumor3D_data/Task08_HepaticVessel_ct/split/',
    '/home/data/tumor3D_data/Task03_Liver_ct/split/',
    '/home/data/tumor3D_data/Task07_Pancreas_ct/split/',
    '/home/data/tumor3D_data/Task10_Colon_ct/split/',
    '/home/data/tumor3D_data/KiPA22/split/',
    '/home/data/tumor3D_data/KiTS23/split/',
    '/home/data/tumor3D_data/TDSC_ABUS/split/',
]

all_classes = [
    'gtvp',
    'gtvn',
    'colon_cancer_primaries',
    'edema',
    'enhancing_tumor',
    'hepatic_tumor',
    'kidney_tumor',
    'liver_tumor',
    'lung_cancer',
    'pancreas_cancer',
    'non_enhancing_tumor',
    'renal_tumor',
    'kidney_tumor',
    'tumor',
    'breast_tumor'
]

all_datasets = [
    'Autopet_ct',
    'Autopet_pet',
    'BraTS_2021_mr_t1',
    'BraTS_2021_mr_t2',
    'BraTS_2021_mr_flair',
    'BraTS_2021_mr_t1ce',
    'HNTSMRG24_mr_t2',
    'Task06_Lung_ct',
    'Task08_HepaticVessel_ct',
    'Task03_Liver_ct',
    'Task07_Pancreas_ct',
    'Task10_Colon_ct',
    'KiPA22',
    'KiTS23',
    'TDSC_ABUS'
]

class_mapping = {
    'Autopet_ct':
        {
            'labels': ['tumor'],
            'class': [1],
            'modality':['computed tomography imaging'],
        },
    'Autopet_pet':
        {
            'labels': ['tumor'],
            'class': [1],
            'modality':['positron emission tomography imaging'],
        },
    'BraTS_2021_mr_t1':
        {
            'labels': ['edema', 'non_enhancing_tumor', 'enhancing_tumor'],
            'class': [2, 1, 4],
            'modality': ['magnetic resonance imaging T1 sequence', 'magnetic resonance imaging T1 sequence', 'magnetic resonance imaging T1 sequence'],

        },
    'BraTS_2021_mr_t2':
        {
            'labels': ['edema', 'non_enhancing_tumor', 'enhancing_tumor'],
            'class': [2, 1, 4],
            'modality': ['magnetic resonance imaging T2 sequence', 'magnetic resonance imaging T2 sequence', 'magnetic resonance imaging T2 sequence'],

        },
    'BraTS_2021_mr_flair':
        {
            'labels': ['edema', 'non_enhancing_tumor', 'enhancing_tumor'],
            'class': [2, 1, 4],
            'modality': ['magnetic resonance imaging Flair sequence', 'magnetic resonance imaging Flair sequence', 'magnetic resonance imaging Flair sequence'],

        },
    'BraTS_2021_mr_t1ce':
        {
            'labels': ['edema', 'non_enhancing_tumor', 'enhancing_tumor'],
            'class': [2, 1, 4],
            'modality': ['magnetic resonance imaging T1 CE sequence', 'magnetic resonance imaging T1 CE sequence', 'magnetic resonance imaging T1 CE sequence'],

        },
    'Task03_Liver_ct':
        {
            'labels': ['liver_tumor'],
            'class': [2],
            'modality': ['computed tomography imaging']
        },
    'Task06_Lung_ct':
        {
            'labels': ['lung_cancer'],
            'class': [1],
            'modality': ['computed tomography imaging'],
        },
    'Task07_Pancreas_ct':
        {
            'labels': ['pancreas_cancer'],
            'class': [2],
            'modality': ['computed tomography imaging'],
        },
    'Task08_HepaticVessel_ct':
        {
            'labels': ['hepatic_tumor'],
            'class': [2],
            'modality': ['computed tomography imaging'],
        },
    'Task10_Colon_ct':
        {
            'labels': ['colon_cancer_primaries'],
            'class': [1],
            'modality': ['computed tomography imaging'],
        },
    'HECKTOR22_ct':
        {
            'labels': ['gtvp', 'gtvn'],
            'class': [1, 2],
            'modality': ['computed tomography imaging', 'computed tomography imaging'],
        },
    'HECKTOR22_pet':
        {
            'labels': ['gtvp', 'gtvn'],
            'class': [1, 2],
            'modality': ['positron emission tomography imaging', 'positron emission tomography imaging'],
        },
    'HNTSMRG24_mr_t2':
        {
            'labels': ['gtvp', 'gtvn'],
            'class': [1, 2],
            'modality': ['magnetic resonance imaging T2 sequence', 'magnetic resonance imaging T2 sequence'],

        },
    'KiPA22':
        {
            'labels': ['renal_tumor'],
            'class': [4],
            'modality':  ['computed tomography angiography imaging'],
        },
    'KiTS23':
        {
            'labels': ['kidney_tumor'],
            'class': [2],
            'modality': ['computed tomography imaging'],
        },
    'TDSC_ABUS':
        {
            'labels': ['breast_tumor'],
            'class': [1],
            'modality': ['ultrasound imaging'],
        },
}
