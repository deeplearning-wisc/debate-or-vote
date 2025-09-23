from datasets import Dataset
from data.data_utils import *

def format_ds(args, tokenizer, model_name, dataset):
    if args.reverse_landmark :
        in_label = 0
    else :
        in_label = 1

    true_dataset = []
    for query, resps in zip(dataset['question'], dataset['correct_answers']):
        for resp in resps:
            true_dataset.append({"input": format_input(args, query, resp, tokenizer, model_name, dialog=True),
                                 "query": format_input(args, query, resp, tokenizer, model_name, dialog=False),
                                 "label": in_label})

    perturbed_dataset = []
    if args.synonym_replacement :
        print("INFO: synonym replacement chosen!")

        for query, resps in zip(dataset['question'], dataset['correct_answers']):
            for resp in resps:
                perturbed_query = replace_with_synonyms(query, args.perturbation)
                perturbed_resp = replace_with_synonyms(resp, args.perturbation)
                perturbed_dataset.append({"input": format_input(args, perturbed_query, perturbed_resp, tokenizer, model_name, dialog=True),
                                          "query": format_input(args, perturbed_query, perturbed_resp, tokenizer, model_name, dialog=False),
                                          "label": 1-in_label})
    elif args.random_deletion :
        print("INFO: random deletion chosen!")

        import pdb;pdb.set_trace()
        for query, resps in zip(dataset['question'], dataset['correct_answers']):
            for resp in resps:
                perturbed_query = random_deletion(query, args.perturbation)
                perturbed_resp = random_deletion(resp, args.perturbation)
                perturbed_dataset.append({"input": format_input(args, perturbed_query, perturbed_resp, tokenizer, model_name, dialog=True),
                                          "query": format_input(args, perturbed_query, perturbed_resp, tokenizer, model_name, dialog=False),
                                          "label": 1-in_label})

    elif args.word_level_shuffling:
        print("INFO: word level response shuffling chosen!")
        for query, resps in zip(dataset['question'], dataset['correct_answers']):
            for resp in resps:
                perturbed_query = shuffle_words_in_sentence(query, args.perturbation)
                perturbed_resp = shuffle_words_in_sentence(resp, args.perturbation)
                perturbed_dataset.append({"input": format_input(args, perturbed_query, perturbed_resp, tokenizer, model_name, dialog=True),
                                          "query": format_input(args, perturbed_query, perturbed_resp, tokenizer, model_name, dialog=False),
                                          "label": 1-in_label})
    elif args.answer_level_shuffling:
        print("INFO: answer shuffling chosen!")
        answers = []
        queries = []
        
        for query, resps in zip(dataset['question'], dataset['correct_answers']):
            for resp in resps:
                # only perturb responses
                queries.append(query)
                answers.append(resp)
        
        shuffled_answers = shuffle_answers(answers, 100)
        for query, perturbed_resp in zip(queries, shuffled_answers):
            perturbed_dataset.append({"input": format_input(args, query, perturbed_resp, tokenizer, model_name, dialog=True),
                                      "query": format_input(args, query, perturbed_resp, tokenizer, model_name, dialog=False),
                                      "label": 1-in_label})
    else :
        print("INFO: True vs False chosen!")
        
        if 'incorrect_answers' not in dataset.keys():
            print("WARNING!!!! True vs. False needs both correct and incorrect responses. Incorrect responses are not available for this dataset!")
        
        if 'incorrect_answers' in dataset.keys():
            for query, resps in zip(dataset['question'], dataset['incorrect_answers']):
                for resp in resps:
                    perturbed_dataset.append({"input": format_input(args, query, resp, tokenizer, model_name, dialog=True), 
                                            "query": format_input(args, query, resp, tokenizer, model_name, dialog=False),
                                            "label": 1-in_label})
    
    if args.reverse_landmark :
        contaminated_dataset = perturbed_dataset + true_dataset
    else :
        contaminated_dataset = true_dataset + perturbed_dataset
    
    dataset = Dataset.from_list(contaminated_dataset)
    return dataset