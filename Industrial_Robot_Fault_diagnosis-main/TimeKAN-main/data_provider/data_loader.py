import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from sktime.datasets import load_from_tsfile_to_dataframe
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Classification(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='label', scale=True, timeenc=0, freq='h',
                 num_classes=None, external_scaler=None):
        # size [seq_len] for classification (no label_len or pred_len needed)
        if size is None:
            self.seq_len = 96
        else:
            self.seq_len = size[0]  # Only use seq_len for classification
        self.stride = 1
        self.label_len = 0
        self.pred_len = 0

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.num_classes = num_classes

        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag

        self.external_scaler = external_scaler
        print(f"[DEBUG] {flag} set: external_scaler is {'provided' if external_scaler is not None else 'None'}")

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler() if self.external_scaler is None else self.external_scaler

        folder_path = os.path.join(self.root_path, self.flag)
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder {folder_path} does not exist: {folder_path}")

        data_files = [f for f in os.listdir(folder_path) if f.endswith('.ts') or f.endswith('.csv')]
        if not data_files:
            raise ValueError(f"No .ts or .csv files found in {folder_path}")

        data_file = data_files[0]
        file_path = os.path.join(folder_path, data_file)

        if data_file.endswith('.ts'):
            sequences, labels = self._parse_ts_file(file_path)
            if sequences is None or len(sequences) == 0:
                raise ValueError(f"Failed to parse TS file {file_path}")
            self.sequences = sequences
            self.labels = np.array(labels).astype(np.int64)
            print(f"DEBUG: Loaded {len(self.sequences)} sequences, each with shape {self.sequences[0].shape}")
        else:
            # CSV path kept for compatibility but you said you use TS
            df_raw = pd.read_csv(file_path)
            cols = list(df_raw.columns)
            if 'date' in cols:
                cols.remove('date')
            label_col = cols[-1]
            feature_cols = cols[:-1]
            if 'date' in df_raw.columns:
                df_features = df_raw[['date'] + feature_cols]
            else:
                df_features = df_raw[feature_cols]

            self.labels = df_raw[label_col].values
            if self.features in ('M', 'MS'):
                if 'date' in df_features.columns:
                    cols_data = [col for col in df_features.columns if col != 'date']
                    df_data = df_features[cols_data]
                else:
                    df_data = df_features
            elif self.features == 'S':
                df_data = df_features[[feature_cols[0]]]

            data_values = df_data.values
            self.sequences = [data_values]
            self.labels = np.array([self.labels[0]]).astype(np.int64)


        if self.scale:
            if self.external_scaler is None:
                all_data = np.vstack(self.sequences)
                self.scaler.fit(all_data)
                self.sequences = [self.scaler.transform(seq) for seq in self.sequences]
            else:
                self.sequences = [self.scaler.transform(seq) for seq in self.sequences]

        self.data_sequences = self.sequences
        self.data_labels = self.labels

        if self.num_classes is None:
            self.num_classes = int(len(np.unique(self.labels)))
        else:
            self.num_classes = int(self.num_classes)

        self.window_indices = []
        for block_idx, seq in enumerate(self.data_sequences):
            T = seq.shape[0]
            if T < self.seq_len:
                print(f"Warning: Block {block_idx} too short ({T} < {self.seq_len}), skipped.")
                continue
            max_start = T - self.seq_len + 1
            for start in range(0, max_start, self.stride):
                self.window_indices.append((block_idx, start))

        print(f"[{self.flag}] Generated {len(self.window_indices)} windows "
              f"from {len(self.data_sequences)} blocks (stride={self.stride})")

    def _parse_ts_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            dimension = 12
            series_length = 256
            data_start = False
            data_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith('@dimension'):
                    parts = line.split()
                    if len(parts) >= 2:
                        dimension = int(parts[1])
                elif line.lower().startswith('@serieslength'):
                    parts = line.split()
                    if len(parts) >= 2:
                        series_length = int(parts[1])
                elif line.lower().startswith('@data'):
                    data_start = True
                    continue
                if data_start:
                    data_lines.append(line)

            sequences = []
            labels = []

            for line in data_lines:
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                data_values = []
                for dim_part in parts[:-1]:
                    dim_values = [float(x.strip()) for x in dim_part.split(',') if x.strip()]
                    data_values.extend(dim_values)

                try:
                    label = float(parts[-1].strip())
                except:
                    continue

                if len(data_values) % dimension != 0:
                    continue

                actual_T = len(data_values) // dimension
                data_array = np.array(data_values).reshape(actual_T, dimension)
                sequences.append(data_array)
                labels.append(int(label))

            return sequences, labels

        except Exception as e:
            print(f"Error parsing TS file {file_path}: {e}")
            return None, None


    def __getitem__(self, index):
        block_idx, start = self.window_indices[index]
        seq_x = self.data_sequences[block_idx][start:start + self.seq_len]  # (seq_len, D)
        label = self.data_labels[block_idx]
        return seq_x, label

    def __len__(self):
        return len(self.window_indices)


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


