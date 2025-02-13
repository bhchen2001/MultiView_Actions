import pandas as pd
import numpy as np

def analyze_dataset(dataset_path):
    # read the dataset
    dataset = pd.read_csv(dataset_path)

    # get the number of samples
    num_samples = len(dataset)
    print(f"Number of samples: {num_samples}")

    # get the instances num of each subject, action, and viewpoint
    num_subjects = len(dataset['subject'].unique())
    num_actions = len(dataset['action'].unique())
    num_viewpoints = len(dataset['viewpoint'].unique())
    print(f"Number of subjects: {num_subjects}")
    print(f"Number of actions: {num_actions}")
    print(f"Number of viewpoints: {num_viewpoints}")

    # get the number of samples per subject, action, and viewpoint
    samples_per_subject = dataset.groupby('subject').size()
    samples_per_action = dataset.groupby('action').size()
    samples_per_viewpoint = dataset.groupby('viewpoint').size()
    print(f"Samples per subject: {samples_per_subject}")
    print(f"Samples per action: {samples_per_action}")
    print(f"Samples per viewpoint: {samples_per_viewpoint}")

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
            for viewpoint in dataset['viewpoint'].unique():
                subset = dataset[(dataset['action'] == action) & (dataset['viewpoint'] == viewpoint)]
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

def generate_balanced_train_dataset(dataset_path, train_dataset_path, test_dataset_path, type, val_ratio=0.2):
    dataset = pd.read_csv(dataset_path)

    # Create an empty DataFrame for the validation dataset
    train_dataset = pd.DataFrame(columns=dataset.columns)
    test_dataset = pd.DataFrame(columns=dataset.columns)

    if type == 'View':
        for subject in dataset['subject'].unique():
            for action in dataset['action'].unique():
                subset = dataset[(dataset['subject'] == subject) & (dataset['action'] == action)]
                if not subset.empty:
                    test_subset = subset.sample(frac=val_ratio, random_state=42)
                    train_subset = subset.drop(test_subset.index)
                    train_dataset = pd.concat([train_dataset, train_subset])
                    test_dataset = pd.concat([test_dataset, test_subset])

    train_dataset.to_csv(train_dataset_path, index=False)
    test_dataset.to_csv(test_dataset_path, index=False)
    print(f"Train dataset saved with {len(train_dataset)} samples.")
    print(f"Test dataset saved with {len(test_dataset)} samples.")


if __name__ == '__main__':
    master_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/NUMAMaster.csv'
    train_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/NUMATrain_View1.csv'
    test_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/NUMATest_View1.csv'
    # val_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/NUMAVal_View1.csv'
    analyze_dataset(master_path)
    # analyze_dataset(test_path)
    # generate_balanced_train_dataset(master_path, train_path, test_path, type='View', val_ratio=0.2)
    # generate_balanced_val_dataset(test_path, val_path, 'View', val_ratio=0.5)
    # analyze_dataset(val_dataset_path)