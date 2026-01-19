import csv
import os

def csv_split_to_ts(
        input_folder: str,
        train_path: str,
        val_path: str,
        test_path: str,
        problem_name: str,
        label_column: str | int = -1,
        selected_columns: list[str | int] | None = None,
        classlabel: str | None = None,
        delimiter: str = ',',
        timestep: int = 35,
        overlap_ratio: float = 0,
        data_ratio: float = 1.0,
):
    csv_files = [
        f for f in os.listdir(input_folder)
        if f.endswith('.csv') and os.path.isfile(os.path.join(input_folder, f))
    ]
    if not csv_files:
        raise ValueError("No CSV files found in the input folder")

    with open(os.path.join(input_folder, csv_files[0]), 'r', encoding='utf-8') as f:
        header = next(csv.reader(f, delimiter=delimiter))

    def resolve_column(col: str | int) -> str:
        return header[col] if isinstance(col, int) else col

    label_col = resolve_column(label_column)
    if label_col not in header:
        raise ValueError(f"Label column {label_col} not found in CSV header")

    if selected_columns is None:
        data_columns = [col for col in header if col != label_col]
    else:
        data_columns = [resolve_column(col) for col in selected_columns if resolve_column(col) != label_col]

    for col in data_columns:
        if col not in header:
            raise ValueError(f"Column {col} not found in CSV header")

    def init_ts_file(path: str, desc_suffix: str):
        with open(path, 'w', encoding='utf-8') as f:
            # 描述块
            f.write(f"# Generated from CSV files in: {input_folder}\n")
            f.write(f"# {desc_suffix} set (contains {os.path.basename(path)})\n")
            f.write(f"# Total source files: {len(csv_files)}\n")
            f.write(f"# Selected columns ({len(data_columns)}): {', '.join(data_columns)}\n")
            f.write(f"@problemName {problem_name}\n")
            f.write("@timeStamps false\n")
            f.write("@missing false\n")
            f.write(f"@univariate false\n")
            f.write(f"@dimesion {len(data_columns)}\n")
            f.write(f"@equallength true\n")
            f.write(f"@serieslength {timestep}\n")
            f.write(f"@targetlabel true\n")
            if classlabel is not None:
                f.write(f"@classlabel true {classlabel}\n")
            else:
                f.write(f"@classlabel false\n")
            f.write("@data\n")

    init_ts_file(train_path, "train")
    init_ts_file(val_path, "validation")
    init_ts_file(test_path, "test")


    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)
            all_rows = list(reader)

            total_rows = len(all_rows)
            selected_rows = int(total_rows * data_ratio)
            all_rows = all_rows[:selected_rows]


            total_len = len(all_rows)
            train_split = int(total_len * 0.7)
            val_split = int(total_len * 0.9)

            train_rows = all_rows[:train_split]
            val_rows = all_rows[train_split:val_split]
            test_rows = all_rows[val_split:]

            overlap_steps = int(timestep * overlap_ratio)

            with open(train_path, 'a', encoding='utf-8') as train_file:
                for i in range(0, len(train_rows) - timestep, timestep - overlap_steps):
                    if len(train_rows) - i < timestep:
                        continue

                    data_segments = [[] for _ in range(len(data_columns))]
                    for t in range(i, i + timestep):
                        row = train_rows[t]
                        for col_idx, col in enumerate(data_columns):
                            data_segments[col_idx].append(row[header.index(col)])

                    label = train_rows[i + timestep - 1][header.index(label_col)]
                    train_file.write(":".join([",".join(seg) for seg in data_segments]) + f":{label}\n")

            with open(val_path, 'a', encoding='utf-8') as val_file:
                for i in range(0, len(val_rows) - timestep, timestep - overlap_steps):
                    if len(val_rows) - i < timestep:
                        continue

                    data_segments = [[] for _ in range(len(data_columns))]
                    for t in range(i, i + timestep):
                        row = val_rows[t]
                        for col_idx, col in enumerate(data_columns):
                            data_segments[col_idx].append(row[header.index(col)])

                    label = val_rows[i + timestep - 1][header.index(label_col)]
                    val_file.write(":".join([",".join(seg) for seg in data_segments]) + f":{label}\n")

            with open(test_path, 'a', encoding='utf-8') as test_file:
                for i in range(0, len(test_rows) - timestep, timestep - overlap_steps):
                    if len(test_rows) - i < timestep:
                        continue

                    data_segments = [[] for _ in range(len(data_columns))]
                    for t in range(i, i + timestep):
                        row = test_rows[t]
                        for col_idx, col in enumerate(data_columns):
                            data_segments[col_idx].append(row[header.index(col)])

                    label = test_rows[i + timestep - 1][header.index(label_col)]
                    test_file.write(":".join([",".join(seg) for seg in data_segments]) + f":{label}\n")


if __name__ == "__main__":
    csv_split_to_ts(
        input_folder="Your path",
        train_path="Your path",
        val_path="Your path",
        test_path="Your path",
        problem_name="SensorActivity",
        label_column="label",
        classlabel="0 1 2 3 4 5 6",
        selected_columns=['vfb0','vfb1','vfb2','vfb3','vfb4','vfb5','current0','current1','current2','current3','current4','current5'],
        timestep=96,
        data_ratio=1,
    )