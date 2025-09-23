from data.base_ds import format_ds
from datasets import load_dataset
import pandas as pd
import re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(full_ans_text: str) -> str:
    match = ANS_RE.search(full_ans_text)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return int(match_str.strip())
    return None

def load_data(args, split='validation'):

    dataset = load_dataset('openai/gsm8k', 'main', cache_dir=args.data_dir)[split]
    dataset = pd.DataFrame(dataset)
    if split == 'train':
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
    else :
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(args.data_size)
    
    questions, labels = [], []
    for question, answer in zip(dataset['question'], dataset['answer']) :
        label = extract_answer(answer)

        questions.append(question)
        labels.append(label)

    return questions, labels