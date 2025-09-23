import pandas as pd
import random
import torch
from datasets import Dataset, concatenate_datasets
import re

def load_data(args, split):
    if args.data == 'arithmetics' :
        from data.arithmetics import load_data as load_arithmetics
        return load_arithmetics(args, split=split)
    elif args.data == 'hellaswag' :
        from data.hellaswag import load_data as load_hellaswag
        return load_hellaswag(args, split=split)
    elif args.data == 'pro_medicine' :
        from data.mmlu_pro_medicine import load_data 
        return load_data(args, split=split)
    elif args.data == 'formal_logic' :
        from data.mmlu_formal_logic import load_data 
        return load_data(args, split=split)
    elif args.data == 'gsm8k' :
        from data.gsm8k import load_data as load_gsm8k
        return load_gsm8k(args, split=split)
    elif args.data == 'csqa' :
        from data.csqa import load_data as load_csqa
        return load_csqa(args, split=split)
    elif args.data == 'hh_rlhf':
        from data.hh_rlhf import load_data as load_hhrlhf
        return load_hhrlhf(args, split=split)
    elif args.data == 'cnn_daily':
        from data.cnn_daily import load_data
        return load_data(args, split=split)
