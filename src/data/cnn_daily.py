
from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, split='validation'):
    split = 'test' if split == 'test' else 'validation'
    dataset = load_dataset('abisee/cnn_dailymail', '3.0.0', cache_dir=args.data_dir)[split]
    dataset = pd.DataFrame(dataset)
    if split == 'train':
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(args.data_size)
    else :
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(args.data_size)

    questions, labels = [], []
    template = 'Summarize the following in three sentences:\n\n"{}"\n\n'
    for ctx, answer in zip(dataset['article'], dataset['highlights']):
        question = template.format(ctx)
        questions.append(question)
        labels.append(answer)
    
    return questions, labels