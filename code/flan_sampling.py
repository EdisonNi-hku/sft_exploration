# Sampling a subset of the full FLAN collection which contains more than 300M instruction-following data
from datasets import load_dataset
from collections import Counter
import pandas as pd


def random_sample(data_num):
    dataset = load_dataset("Open-Orca/FLAN", split='train', streaming=True)
    shuffled_dataset = dataset.shuffle(seed=42)
    subset_size = data_num
    subset = shuffled_dataset.take(subset_size)
    df = pd.DataFrame(list(subset))
    grouped = df.groupby('_task_name').size()
    print(grouped)
    print("Min:", grouped.min())
    print("Max:", grouped.max())
    df.to_csv("flan_sampled_" + str(data_num) + '.csv')


def task_name_count():
    task_name_counts = Counter()
    dataset_stream = load_dataset("Open-Orca/FLAN", split='train', streaming=True, batch_size=10000,
                                  batched=True)

    # Iterate over the dataset by batches to count _task_name values
    for batch in dataset_stream:
        # Update counts for each _task_name in the batch
        task_names = batch['_task_name']
        task_name_counts.update(task_names)

    # After iterating through the dataset, task_name_counts will contain the counts of each unique _task_name
    print(task_name_counts)
    # If you just want to know the unique values (and not their counts), you can use:
    unique_task_names = set(task_name_counts.keys())
    print(unique_task_names)


if __name__ == '__main__':
    random_sample(100000)
