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
    elif type == 'View':
        for subject in dataset['subject'].unique():
            for action in dataset['action'].unique():
                subset = dataset[(dataset['subject'] == subject) & (dataset['action'] == action)]
                if not subset.empty:
                    train_subset = subset.sample(frac=1-val_ratio, random_state=42)
                    test_subset = subset.drop(train_subset.index)
                    train_dataset = pd.concat([train_dataset, train_subset])
                    test_dataset = pd.concat([test_dataset, test_subset])

    elif type == 'CV':
        # set test set as camera 1, others as train set
        train_dataset = dataset[dataset['camera'] != 4]
        test_dataset = dataset[dataset['camera'] == 4]

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

def generated_balanced_k_fold_dataset(master_path, fold_paths, kfold, random_state = 42):
    dataset = pd.read_csv(master_path)

    for subject in dataset['camera'].unique():
        subset_idx = dataset[(dataset['camera'] == subject)].index.tolist()
        if not subset_idx:
            continue
        print("Length of subset: ", len(subset_idx))
        np.random.seed(random_state)
        np.random.shuffle(subset_idx)
        folds = np.array_split(subset_idx, kfold)
        for fold_idx, indices in enumerate(folds):
            dataset.loc[indices, 'fold'] = fold_idx

    # print each fold's len
    for k in range(kfold):
        print(f"Fold {k} has {len(dataset[dataset['fold'] == k])} samples.")

    dataset['subject'] = dataset['subject'].apply(lambda x: '{0:0>3}'.format(x))
    dataset['action'] = dataset['action'].apply(lambda x: '{0:0>3}'.format(x))
    dataset['camera'] = dataset['camera'].apply(lambda x: '{0:0>3}'.format(x))
    dataset['setup'] = dataset['setup'].apply(lambda x: '{0:0>3}'.format(x))
    dataset['subject'] = dataset['subject'].apply(lambda x: '{0:0>3}'.format(x))
    dataset['repetition'] = dataset['repetition'].apply(lambda x: '{0:0>3}'.format(x))

    for k in range(kfold):
        fold_train_path = dataset[dataset['fold'] != k].copy()
        fold_train_path.drop(columns=['fold'], inplace=True)
        fold_train_path.to_csv(fold_paths[k] + '_train.csv', index=False)

        fold_test_path = dataset[dataset['fold'] == k].copy()
        fold_test_path.drop(columns=['fold'], inplace=True)
        fold_test_path.to_csv(fold_paths[k] + '_test.csv', index=False)

        print(f"Fold {k} train dataset saved with {len(fold_train_path)} samples.")
        print(f"Fold {k} test dataset saved with {len(fold_test_path)} samples.")

def generate_rc_pair_dataset(master_path, train_path, test_path):
    dataset = pd.read_csv(master_path)

    train_dataset = []
    test_dataset = []

    r_c_pairs = {
        1: [3],
        3: [4],
        4: [3, 4],
        5: [4],
        6: [4],
        7: [1, 3],
        8: [1, 2],
        9: [2, 4],
        10: [3],
        11: [3],
        12: [3, 4],
        13: [3],
        15: [4],
        16: [3, 4],
        17: [4],
        18: [4],
        19: [1, 3],
        20: [1, 2],
        21: [2, 4],
        22: [3],
        23: [3],
        24: [3, 4],
    }

    # iterate master list test match rc pairs, others train
    for idx, row in dataset.iterrows():
        if row['repetition'] in r_c_pairs.keys() and row['camera'] in r_c_pairs[row['repetition']]:
            test_dataset.append(row)
        else:
            train_dataset.append(row)

    train_dataset = pd.DataFrame(train_dataset)
    test_dataset = pd.DataFrame(test_dataset)

    train_dataset.to_csv(train_path, index=False)
    test_dataset.to_csv(test_path, index=False)

    print(f"Train dataset saved with {len(train_dataset)} samples.")
    print(f"Test dataset saved with {len(test_dataset)} samples.")

def generated_balanced_k_fold_dataset_S003(master_path, fold_paths, kfold, random_state=42):

    # 讀取主檔案
    dataset = pd.read_csv(master_path)

    # 以 subject, action, repetition, setup 四個欄位分群
    groups = dataset.groupby(['subject', 'action', 'repetition', 'setup']).groups
    unique_groups = list(groups.keys())

    # 設定隨機種子並隨機排列群組順序
    np.random.seed(random_state)
    np.random.shuffle(unique_groups)

    # 將群組依照 k 折劃分，每折包含若干群組
    folds = np.array_split(unique_groups, kfold)
    for fold_idx, group_keys in enumerate(folds):
        for key in group_keys:
            indices = groups[key]
            dataset.loc[indices, 'fold'] = fold_idx

    # 印出各折的樣本數
    for k in range(kfold):
        print(f"Fold {k} has {len(dataset[dataset['fold'] == k])} samples.")

    # 對指定欄位進行格式化處理（例如補零至 3 位數）
    dataset['subject'] = dataset['subject'].apply(lambda x: '{0:0>3}'.format(x))
    dataset['action'] = dataset['action'].apply(lambda x: '{0:0>3}'.format(x))
    dataset['camera'] = dataset['camera'].apply(lambda x: '{0:0>3}'.format(x))
    dataset['setup'] = dataset['setup'].apply(lambda x: '{0:0>3}'.format(x))
    dataset['repetition'] = dataset['repetition'].apply(lambda x: '{0:0>3}'.format(x))

    # 儲存每折的訓練集與測試集
    for k in range(kfold):
        fold_train = dataset[dataset['fold'] != k].copy()
        fold_train.drop(columns=['fold'], inplace=True)
        fold_train.to_csv(fold_paths[k] + '_train.csv', index=False)

        fold_test = dataset[dataset['fold'] == k].copy()
        fold_test.drop(columns=['fold'], inplace=True)
        fold_test.to_csv(fold_paths[k] + '_test.csv', index=False)

        print(f"Fold {k} train dataset saved with {len(fold_train)} samples.")
        print(f"Fold {k} test dataset saved with {len(fold_test)} samples.")


if __name__ == '__main__':
    base_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/lab639_fisheye/'
    master_path = base_path + 'fisheye639_S003_master.csv'
    # train_path = base_path + 'fisheye639_S002_train.csv'
    # test_path = base_path + 'fisheye639_S002_test.csv'

    train_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/lab639_fisheye/fisheye639_S003_CV4_train.csv'
    test_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/lab639_fisheye/fisheye639_S003_CV4_test.csv'

    generate_balanced_train_dataset(master_path, train_path, test_path, type='CV')
    # generate_rc_pair_dataset(master_path, train_path, test_path)

    # fold_paths = [
    #     base_path + 'fisheye639_S003_fold{}'.format(i) for i in range(5)
    # ]
    # generated_balanced_k_fold_dataset(master_path, fold_paths, 5)

    # analyze_dataset(master_path)
    # analyze_dataset(train_path)
    # analyze_dataset(test_path)
    # generate_balanced_train_dataset(master_path, train_path, test_path, type='Equal', val_ratio=0.2, view = 4)
    # generate_balanced_val_dataset(test_path, val_path, 'View', val_ratio=0.5)
    # analyze_dataset(val_dataset_path)