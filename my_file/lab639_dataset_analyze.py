import pandas as pd
import numpy as np

def analyze_dataset(dataset_path, view = -1, subject = -1, action = -1):
    # read the dataset
    dataset = pd.read_csv(dataset_path)

    if view > 0:
        dataset = dataset[dataset['camera'] == view]

    # get the number of samples
    num_samples = len(dataset)
    print(f"Number of samples: {num_samples}")

    # get the instances num of each subject, action, and camera
    num_subjects = len(dataset['subject'].unique())
    num_actions = len(dataset['action'].unique())
    num_viewpoints = len(dataset['camera'].unique())
    num_setup = len(dataset['setup'].unique())
    print(f"Number of subjects: {num_subjects}")
    print(f"Number of actions: {num_actions}")
    print(f"Number of viewpoints: {num_viewpoints}")
    print(f"Number of setups: {num_setup}")

    # get the number of samples per subject, action, and camera
    samples_per_subject = dataset.groupby('subject').size()
    samples_per_action = dataset.groupby('action').size()
    samples_per_viewpoint = dataset.groupby('camera').size()
    samples_per_setup = dataset.groupby('setup').size()
    print(f"Samples per subject: {samples_per_subject}")
    print(f"Samples per action: {samples_per_action}")
    print(f"Samples per camera: {samples_per_viewpoint}")
    print(f"Samples per setup: {samples_per_setup}")

def generate_balanced_val_dataset(dataset_path, val_dataset_path, type, val_ratio):
    dataset = pd.read_csv(dataset_path)

    # Create an empty DataFrame for the validation dataset
    val_dataset = pd.DataFrame(columns=dataset.columns)

    if type == 'CV':
        for subject in dataset['subject'].unique():
            for action in dataset['action'].unique():
                subset = dataset[(dataset['subject'] == subject) & (dataset['action'] == action)]
                if not subset.empty:
                    val_subset = subset.sample(frac=val_ratio, random_state=42)
                    val_dataset = pd.concat([val_dataset, val_subset])
    elif type == 'CS':
        for action in dataset['action'].unique():
            for camera in dataset['camera'].unique():
                subset = dataset[(dataset['action'] == action) & (dataset['camera'] == camera)]
                if not subset.empty:
                    val_subset = subset.sample(frac=val_ratio, random_state=42)
                    val_dataset = pd.concat([val_dataset, val_subset])
    elif type == 'View':
        for subject in dataset['subject'].unique():
            for action in dataset['action'].unique():
                subset = dataset[(dataset['subject'] == subject) & (dataset['action'] == action)]
                print(f"Subset: {len(subset)}")
                if not subset.empty:
                    val_subset = subset.sample(frac=val_ratio, random_state=42)
                    val_dataset = pd.concat([val_dataset, val_subset])

    # Save the validation dataset
    val_dataset.to_csv(val_dataset_path, index=False)
    print(f"Validation dataset saved with {len(val_dataset)} samples.")

def generate_balanced_train_dataset(dataset_path, train_dataset_path, test_dataset_path, type, val_ratio=0.2, view = 0):
    dataset = pd.read_csv(dataset_path)

    if type == 'View' and view > 0:
        dataset = dataset[dataset['camera'] == view]
    elif type == 'View' and view <=0:
        raise ValueError("View should be greater than 0 for View type dataset.")
    
    train_dataset = pd.DataFrame()
    test_dataset = pd.DataFrame()

    # for each action and camera, split the dataset into train and test sets with ratio
    if type == 'Equal':
        for subject in dataset['subject'].unique():
            for action in dataset['action'].unique():
                subset = dataset[(dataset['subject'] == subject) & (dataset['action'] == action)]
                if not subset.empty:
                    train_subset = subset.sample(frac=1-val_ratio, random_state=42)
                    test_subset = subset.drop(train_subset.index)
                    train_dataset = pd.concat([train_dataset, train_subset])
                    test_dataset = pd.concat([test_dataset, test_subset])

    # convert numeric columns to 3-digits format (17 --> 017)
    train_dataset['subject'] = train_dataset['subject'].apply(lambda x: '{0:0>3}'.format(x))
    train_dataset['action'] = train_dataset['action'].apply(lambda x: '{0:0>3}'.format(x))
    train_dataset['camera'] = train_dataset['camera'].apply(lambda x: '{0:0>3}'.format(x))
    train_dataset['setup'] = train_dataset['setup'].apply(lambda x: '{0:0>3}'.format(x))
    train_dataset['repetition'] = train_dataset['repetition'].apply(lambda x: '{0:0>3}'.format(x))

    test_dataset['subject'] = test_dataset['subject'].apply(lambda x: '{0:0>3}'.format(x))
    test_dataset['action'] = test_dataset['action'].apply(lambda x: '{0:0>3}'.format(x))
    test_dataset['camera'] = test_dataset['camera'].apply(lambda x: '{0:0>3}'.format(x))
    test_dataset['setup'] = test_dataset['setup'].apply(lambda x: '{0:0>3}'.format(x))
    test_dataset['repetition'] = test_dataset['repetition'].apply(lambda x: '{0:0>3}'.format(x))

    train_dataset.to_csv(train_dataset_path, index=False)
    test_dataset.to_csv(test_dataset_path, index=False)
    print(f"Train dataset saved with {len(train_dataset)} samples.")
    print(f"Test dataset saved with {len(test_dataset)} samples.")


if __name__ == '__main__':
    master_path = '/mnt/disk1/bhchen/action_recognition/dataset/lab639_fisheye/fisheye639_partial_master.csv'
    train_path = '/mnt/disk1/bhchen/action_recognition/dataset/lab639_fisheye/fisheye639_partial_train.csv'
    test_path = '/mnt/disk1/bhchen/action_recognition/dataset/lab639_fisheye/fisheye639_partial_test.csv'
    # train_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/ori/NTU60Train_CSmap.csv'
    # test_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/ori/NTU60Test_CSmap.csv'
    # val_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/NUMAVal_View1.csv'
    # analyze_dataset(master_path)
    analyze_dataset(train_path)
    analyze_dataset(test_path)
    # generate_balanced_train_dataset(master_path, train_path, test_path, type='Equal', val_ratio=0.2)
    # generate_balanced_val_dataset(test_path, val_path, 'View', val_ratio=0.5)
    # analyze_dataset(val_dataset_path)