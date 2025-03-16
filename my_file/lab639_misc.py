import os
import pandas as pd
import timeit
import cv2
import numpy as np
import torch
import h5py
import math
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
    Resize,
    PILToTensor
)

def create_master_dataset_csv_S003():
    dataset_base_dir = '/mnt/disk1/bhchen/action_recognition/dataset/lab639_fisheye'
    anno_base_dir = '/home/bhchen/action_recognition/dataset/lab639_fisheye'
    frame_anno = anno_base_dir + '/avi_start_end_S003.csv'

    master_csv_dir = anno_base_dir + '/fisheye639_S003_master.csv'

    viewpoint_num = 4

    # create a csv file with following columns: video_id, subject, action, camera, repetition, setup
    # format in anno_base: SxxxPxxxRxxxAxxx
    # extend format for : SxxxCxxxPxxxRxxxAxxx, Cxxx is the camera number, from 1 to viewpoint_num

    anno = pd.read_csv(frame_anno)
    master_df_rows = []

    for i, row in anno.iterrows():
        setup = row['data'][:4]
        subject = row['data'][4:8]
        repetition = row['data'][8:12]
        action = row['data'][12:16]

        for camera in range(1, viewpoint_num + 1):
            video_id = setup + 'C' + str(camera).zfill(3) + subject + repetition + action
            master_df_rows.append([video_id, subject[1:], action[1:], str(camera).zfill(3), repetition[1:], setup[1:]])

    # sort the rows with the video_id
    master_df_rows.sort(key=lambda x: x[0])

    # write the rows to the csv file
    with open(master_csv_dir, 'w') as f:
        f.write('video_id,subject,action,camera,repetition,setup\n')
        for row in master_df_rows:
            f.write(','.join(row) + '\n')

    return


def create_fisheye_dataset_csv():
    # get the file names with corresponding R-C pairs
    # when A = 1, 2, 3, 5, 6, 7, 8, 9, 10:
        # when R = 1, 6, 11, 16, C = 1, 2
        # when R = 2, 7, 12, 17, C = 2, 3
        # when R = 3, 8, 13, 18, C = 2, 3, 4
        # when R = 4, 9, 14, 19, C = 2, 3, 4
        # when R = 5, 10, 15, 20, C = 1, 2, 3, 4
        # when R = 21, 22, 23, C = 1, 2, 3, 4
        # when R = 24, C = 1, 2, 3
    # list all the file names in the directory in this format:
    # Axxx/SxxxPxxxRxxxAxxx.avi
    # for example, the condition of A = 1, R = 1, C = 1, the file name format is A001/S001P001R001A001.avi
    # the base directory is /mnt/disk1/bhchen/action_recognition/dataset/lab639_fisheye

    # create a csv file with the following columns: video_id (file name), subject (P), action (A), camera (C), repetition (R), setup (S)

    base_dir = '/mnt/disk1/bhchen/action_recognition/dataset/lab639_fisheye'
    dataset_csv_dir = '/home/bhchen/action_recognition/dataset/lab639_fisheye'
    R_C_pairs = {
        # create the pair dictionary with the corresponding R-C pairs
        1: [1, 2],
        2: [2, 3],
        3: [2, 3, 4],
        4: [2, 3, 4],
        5: [1, 2, 3, 4],
        6: [1, 2],
        7: [2, 3],
        8: [2, 3, 4],
        9: [2, 3, 4],
        10: [1, 2, 3, 4],
        11: [1, 2],
        12: [2, 3],
        13: [2, 3, 4],
        14: [2, 3, 4],
        15: [1, 2, 3, 4],
        16: [1, 2],
        17: [2, 3],
        18: [2, 3, 4],
        19: [2, 3, 4],
        20: [1, 2, 3, 4],
        21: [1, 2, 3, 4],
        22: [1, 2, 3, 4],
        23: [1, 2, 3, 4],
        24: [1, 2, 3]
    }
    # train_df_rows = []
    # test_df_rows = []
    master_df_rows = []
    for folder in os.listdir(base_dir):
        # exclude /example/ folder
        # if folder not in format 'Axxx', continue
        if 'A0' not in folder or folder == 'A000':
            continue
        for video in os.listdir(os.path.join(base_dir, folder)):
            if 'S002' in video:
                setup = video[0:4]
                camera = video[4:8]
                subject = video[8:12]
                repetition = video[12:16]
                action = video[16:20]
                master_df_rows.append([video, subject[1:], action[1:], camera[1:], repetition[1:], setup[1:]])
                # if int(action[1:]) == 4:
                #     continue
                # print(video, subject, action, repetition, camera, setup)
                # if int(repetition[1:]) in R_C_pairs.keys() and int(camera[1:]) in R_C_pairs[int(repetition[1:])]:
                    # print(video)
                    # master_df_rows.append([video, subject[1:], action[1:], camera[1:], repetition[1:], setup[1:]])
        
        # sort the rows with the video_id
        master_df_rows.sort(key=lambda x: x[0])

        # write the rows to the csv file
        with open(dataset_csv_dir + '/fisheye639_S003_master.csv', 'w') as f:
            f.write('video_id,subject,action,camera,repetition,setup\n')
            for row in master_df_rows:
                f.write(','.join(row) + '\n')
    return

def create_sampled_jpg_from_avi():
    base_dir = '/mnt/disk1/bhchen/action_recognition/dataset/lab639_fisheye'
    anno = pd.read_csv(base_dir + '/fisheye639_partial_master.csv')
    resize = Resize([270, 480])
    frames = []
    for i, video in enumerate(anno['video_id']):
        start = timeit.default_timer()
        # if i % 100 == 0:
        #     print(i, flush=True)
        if os.path.exists(base_dir + '/sampled_jpg/' + video):
            print(base_dir + '/sampled_jpg/' + video + 'already exists!', flush=True)
            continue
        count = 0
        if ".avi" in video:
            start = timeit.default_timer()
            # video_path is base_dir + /Axxx/ + video
            video_path = os.path.join(base_dir, video[-8:-4], video)
            # print(video_path)
            # continue
            cap = cv2.VideoCapture(os.path.join(base_dir, video))

            if not cap.isOpened():
                print(f"Failed to open {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, num=32, dtype=int)
            sampled_frames = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    sampled_frames.append(frame)

            cap.release()

            # save the sampled frames to the directory
            os.makedirs(base_dir + '/sampled_jpg/' + video, exist_ok=True)
            for i, frame in enumerate(sampled_frames):
                cv2.imwrite(base_dir + '/sampled_jpg/' + video + '/' + str(i) + '.jpg', frame)
            stop = timeit.default_timer()
            print(f"Video {video} done in {stop - start}s")
        break
    return

def create_sampled_jpg(sampled_frames_num = 32):
    base_dir = '/mnt/disk1/bhchen/action_recognition/dataset/lab639_fisheye'
    anno = pd.read_csv(base_dir + '/fisheye639_partial_master.csv')
    start_end_anno = pd.read_csv(base_dir + '/avi_start_end.csv')

    # select the rows that video_id contains 'S002'
    anno = anno[anno['video_id'].str.contains('S002')]

    resize = Resize([270, 480])
    frames = []
    for i, video in enumerate(anno['video_id']):

        jpg_path = os.path.join(base_dir, video[-8:-4], video[:-4])
        jpg_files = os.listdir(jpg_path)

        # get the start and end frame of the video
        # check if the video is in the start_end_anno
        data_name = video[:4] + video[8:-4]
        if data_name not in start_end_anno['data'].values:
            print(f"{data_name} not in the start_end_anno")
            break
        start_frame = start_end_anno[start_end_anno['data'] == data_name]['start'].values
        end_frame = start_end_anno[start_end_anno['data'] == data_name]['end'].values

        print(f"Video: {video}, start: {start_frame}, end: {end_frame}")

        # get frames b/w start and end frame
        frame_indices = np.linspace(start_frame, end_frame, num=sampled_frames_num, dtype=int)
        sampled_frames = []

        for idx in frame_indices:
            # turn idx to format xxxx.jpg, for example, 0001.jpg, first remove '[' and ']', then zfill(4)
            idx = str(idx).replace('[', '').replace(']', '').zfill(4) + '.jpg'
            # print(idx)
            if idx not in jpg_files:
                print(f"{idx} not in {jpg_path}")
                break
            frame = cv2.imread(os.path.join(jpg_path, idx))
            sampled_frames.append(frame)

        saved_path = os.path.join(base_dir, 'sampled_jpg', video[:-4])
        # print(saved_path)
        if os.path.exists(saved_path):
            print(f"{saved_path} already exists!")
            break
        os.makedirs(saved_path)

        # save the sampled frames to the directory
        for i, frame in enumerate(sampled_frames):
            cv2.imwrite(os.path.join(saved_path, str(i).zfill(4) + '.jpg'), frame)
        print(f"Video {video} done")

    return

def create_h5py_old():
    base_dir = '/mnt/disk1/bhchen/action_recognition/dataset/lab639_fisheye/sampled_jpg'
    save_dir = '/mnt/disk1/bhchen/action_recognition/dataset/lab639_fisheye/processed_data'
    resize = Resize([270, 480])
    # create h5py file with all jpg files in folder
    for folders in sorted(os.listdir(base_dir)):
        frames = []
        for jpg in sorted(os.listdir(os.path.join(base_dir, folders))):
            jpg_path = os.path.join(base_dir, folders, jpg)
            # print(jpg_path)
            frame = cv2.imread(jpg_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.as_tensor(frame)
            frame = frame.permute(2, 0, 1)
            frame = resize(frame)
            frames.append(frame)
        tframes = torch.stack([frame for frame in frames])
        frames.clear()

        # save the tensor to h5py file
        with h5py.File(os.path.join(save_dir, folders + '.avi.hdf5'), 'w') as f:
            f.create_dataset('default', data=tframes)

        del tframes
        print(f"{folders} done")
    return

def create_h5py(select_setup, sampled_frames_num):
    dataset_base_dir = '/mnt/disk1/bhchen/action_recognition/dataset/lab639_fisheye'
    dataset_csv_dir = '/home/bhchen/action_recognition/dataset/lab639_fisheye'
    dataset_save_dir = '/mnt/disk1/bhchen/action_recognition/dataset/lab639_fisheye/processed_data_S003'

    # select the setup according to user
    master_anno = pd.read_csv(dataset_csv_dir + '/fisheye639_S003_master.csv')
    if select_setup != 'all':
        master_anno = master_anno[master_anno['video_id'].str.contains(select_setup)]

    start_end_anno = pd.read_csv(dataset_csv_dir + '/avi_start_end_S003.csv')
    if select_setup != 'all':
        start_end_anno = start_end_anno[start_end_anno['data'].str.contains(select_setup)]

    resize = Resize([270, 480])

    flag = False

    for i, video_id in enumerate(master_anno['video_id']):

        if flag == False and video_id == 'S003C004P003R011A005':
            flag = True
        elif flag == False:
            print("Skip : {}".format(video_id))
            continue

        data_name = video_id[:4] + video_id[8:]
        if data_name not in start_end_anno['data'].values:
            print(f"{data_name} not in the start_end_anno")
            break
        start_frame = start_end_anno[start_end_anno['data'] == data_name]['start'].values
        end_frame = start_end_anno[start_end_anno['data'] == data_name]['end'].values

        print(f"Video_id: {video_id}, start: {start_frame}, end: {end_frame}")

        # get frames b/w start and end frame
        frame_indices = np.linspace(start_frame, end_frame, num=sampled_frames_num, dtype=int)
        sampled_frames = []

        action_class = video_id[-4:]
        repetition = int(video_id[-7:-4])

        if action_class == 'A006':
            action_dir = 'A005'
        elif action_class == 'A008':
            action_dir = 'A007'
        else:
            action_dir = action_class

        if action_class != 'A004':
            video_folder = video_id[:12] + 'R' + str((repetition - 1) // 4 * 4 + 1).zfill(3) + action_dir
        elif action_class == 'A004':
            video_folder = video_id[:12] + 'R' + str((repetition - 1) // 3 * 3 + 1).zfill(3) + action_dir
        
        jpg_path = os.path.join(dataset_base_dir, action_dir, video_folder)

        print("Video_id: {}, jpg_path: {}".format(video_id, jpg_path))

        jpg_files = os.listdir(jpg_path)

        for idx in frame_indices:
            # turn idx to format xxxx.jpg, for example, 0001.jpg, first remove '[' and ']', then zfill(4)
            idx = str(idx).replace('[', '').replace(']', '').zfill(4) + '.jpg'
            # print(idx)
            if idx not in jpg_files:
                print(f"{idx} not in {jpg_path}")
                break
            frame = cv2.imread(os.path.join(jpg_path, idx))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.as_tensor(frame)
            frame = frame.permute(2, 0, 1)
            frame = resize(frame)
            sampled_frames.append(frame)
        tframes = torch.stack([frame for frame in sampled_frames])
        sampled_frames.clear()

        # save the tensor to h5py file
        with h5py.File(os.path.join(dataset_save_dir, video_id + '.hdf5'), 'w') as f:
            f.create_dataset('default', data=tframes)

        del tframes
        print(f"{video_id} done")
    return


if __name__ == '__main__':
    # create_master_dataset_csv_S003()
    # create_fisheye_dataset_csv()
    # create_sampled_jpg(16)
    create_h5py('S003', 16)