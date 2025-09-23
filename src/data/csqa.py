
from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd

def load_data(args, split='validation'):
    split = 'validation' if split == 'test' else split
    dataset = load_dataset('tau/commonsense_qa', cache_dir=args.data_dir)[split]
    dataset = pd.DataFrame(dataset)
    if split == 'train':
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(args.data_size)
    else :
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(300)

    questions, labels = [], []
    choices = "ABCDE"
    template = '{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n(E) {}\n\n'
    for ctx, choices, answer in zip(dataset['question'], dataset['choices'], dataset['answerKey']):
        options = choices['text']
        if len(options) != 5 :
            continue
        question = template.format(ctx, options[0], options[1], options[2], options[3], options[4])
        label = f"({answer})"
        questions.append(question)
        labels.append(label)
    
    return questions, labels