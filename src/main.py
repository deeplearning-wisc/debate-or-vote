
import argparse, sys, os, copy, time, random, json, pickle, re, collections, gc
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import torch
from rouge_score import rouge_scorer
ROUGE = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])


from model.model_utils import get_agents, engine
from data.data_utils import load_data
from evaluator import get_instruction_suffix, evaluate_arithmetics, evaluate_mcq, base_evaluate_arithmetics, base_evaluate_mcq, evaluate_gen



def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serializable")


def get_args():

    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default="out/")

    # data
    parser.add_argument('--data_dir', type=str, default="/nobackup2/froilan/datasets/")
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--sub_data', type=str, default='')
    parser.add_argument('--data_size', type=int, default=0)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--debug', action='store_true')

    # agent
    parser.add_argument('--num_agents', type=int, default=5)

    parser.add_argument('--agent_selection', type=str, default="none")
    parser.add_argument('--multi_persona', action='store_true')


    # model
    parser.add_argument('--model', type=str, default="llama3.1")
    parser.add_argument('--model_dir', type=str, default="/nobackup2/froilan/models/")
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')


    # debate
    parser.add_argument('--debate_rounds', type=int, default=5)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--centralized', action='store_true')

    parser.add_argument('--solver', type=str, default='vote', choices=['vote','debate'])
    parser.add_argument('--generate_first_round', action='store_true')
    parser.add_argument('--max_num_agents', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--bae', action='store_true', help="base answer extractor")
    parser.add_argument('--cot', action='store_true')


    return parser.parse_args()


def get_new_message(args, sample, responses, personas=None, suffix=None):

    new_message = {}

    agents = list(responses.keys())
    if len(agents) > 1 : # MULTI-AGENT DEBATE

        if not args.centralized : # DECENTRALIZED MAD
            for i, agent in enumerate(agents) :
                msg = "These are the recent opinions from other agents: "
                if args.sparse :
                    peers = [agents[(i-1) % len(agents)], agents[(i+1) % len(agents)]]
                else :
                    peers = agents[:i]+agents[i+1:]
                for other_agent in peers:
                    msg += f"\n\nOne of the agents' response: \n{responses[other_agent]}\n"
                msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
                msg += f'\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}'

                if suffix is not None :
                    msg += suffix

                if personas is not None :
                    new_message[agent] = [{'role': 'system', 'content': personas[agent.split("__")[-2]]},{'role': 'user', 'content': msg}]
                else :
                    new_message[agent] = {'role': 'user', 'content': msg}

        else : # CENTRALIZED MAD
            for i, agent in enumerate(agents):
                if i == 0 :
                    msg = "These are the recent opinions from other agents: "
                    peers = agents[:i]+agents[i+1:]
                    for other_agent in peers:
                        msg += f"\n\nOne of the agents' response: \n{responses[other_agent]}\n"
                    msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
                    msg += f'\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}'
                else :
                    msg = f"This is the recent opinion from another agent: \n{responses[agents[0]]}\n"
                    msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
                    msg += f'\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}'
                
                if suffix is not None :
                    msg += suffix

                if personas is not None :
                    new_message[agent] = [{'role': 'system', 'content': personas[agent.split("__")[-2]]},{'role': 'user', 'content': msg}]
                else :
                    new_message[agent] = {'role': 'user', 'content': msg}

    else : # SINGLE AGENT SELF REFINEMENT
        for i, agent in enumerate(agents) :
            msg = f"This was your most recent opinion:\n{responses[agents[i]]}\n"
            msg += f'\n\nRevise your recent opinion to give your updated final answer to the question:\n{sample}'

            if suffix is not None :
                msg += suffix

            if personas is not None :
                new_message[agent] = [{'role': 'system', 'content': personas[agent.split("__")[-2]]},{'role': 'user', 'content': msg}]
            else :
                new_message[agent] = {'role': 'user', 'content': msg}

    return new_message


def main(args):

    # Load Agents
    agent, personas = get_agents(args)
      
    # Load Data
    test_X, test_Y = load_data(args, split='test')


    # Setup Names
    fname = f"{args.data}_{args.data_size}__{args.model}_N={args.num_agents}_R={args.debate_rounds}"
    if args.sparse : fname += '_SPARSE'
    elif args.centralized : fname += '_CENTRAL'
    if args.bae : fname += '_BAE'
    if args.multi_persona : fname += '_HETERO'

    agent_names = []
    for i in range(args.num_agents):
        for persona in personas.keys():
            agent_names.append(f"{args.data}_{args.data_size}__{args.model}__{persona}__Agent{i+1}")
          

    # Setup Experiments
    SUFFIX = get_instruction_suffix(args)

    if args.data in ['arithmetics','gsm8k']:
        if args.bae :
            evaluate = base_evaluate_arithmetics
        else :
            evaluate = evaluate_arithmetics
    elif args.data in ['hellaswag','pro_medicine','formal_logic','csqa','hh_rlhf']:
        if args.bae:
            evaluate = base_evaluate_mcq
        else :
            evaluate = evaluate_mcq
    elif args.data in ['cnn_daily'] :
        evaluate = evaluate_gen
    else :
        raise NotImplementedError

    
    # Debate
    sample_responses = []
    iscorr_list = []


    for i, (x, y) in tqdm(enumerate(zip(test_X, test_Y)), total=len(test_X)):

        print('\n\nQuestion: ', x + SUFFIX, '\n')

        # initialize opinions
        print("Gathering initial opinions...")
        round_iscorr = []
        if args.multi_persona :
            messages = []
            for name, sys in personas.items():
                messages.append([{"role": "system", "content": sys},{"role": "user", "content": x + SUFFIX}])
        else:
            messages = [{"role": "user", "content": x + SUFFIX}] * args.num_agents
        responses = engine(messages, agent, args.num_agents)
        agent_responses = dict(zip(agent_names, responses))

        # evaluate
        if args.centralized :
            central_agent_response = {list(agent_responses.keys())[0] : list(agent_responses.values())[0]}
            final_resps, debate_resps, is_corr = evaluate(central_agent_response, y)
        else :
            final_resps, debate_resps, is_corr = evaluate(agent_responses, y)

        print(f"ROUND 0 : {final_resps} (answer = {y})")
        if args.data in ['arithmetics','gsm8k']:
            round_data = {
                'responses': agent_responses,
                'final_answers': final_resps,
                'final_answer_iscorr': [y_pred == np.round(y,1) for y_pred in final_resps],
                'debate_answer': debate_resps,
                'debate_answer_iscorr': is_corr,
                'answer': np.round(y, 1),
            }
        else :
            round_data = {
                'responses': agent_responses,
                'final_answers': final_resps,
                'final_answer_iscorr': [y_pred == y for y_pred in final_resps],
                'debate_answer': debate_resps,
                'debate_answer_iscorr': is_corr,
                'answer': y,
            }
        rounds_data_dict = {'0': round_data}
        round_iscorr.append(is_corr)

        start = 1


        # begin debate
        for r in range(start, args.debate_rounds+1) :

            print(f"Debating round {r}...")
            if args.multi_persona:
                new_agent_messages = get_new_message(args, x, agent_responses, personas, suffix=SUFFIX)
            else:
                new_agent_messages = get_new_message(args, x, agent_responses, suffix=SUFFIX)
            messages = list(new_agent_messages.values())
            responses = engine(messages, agent, args.num_agents)
            agent_responses = dict(zip(agent_names, responses))

            # evaluate
            if args.centralized:
                central_agent_response = {list(agent_responses.keys())[0] : list(agent_responses.values())[0]}
                final_resps, debate_resps, is_corr = evaluate(central_agent_response, y)
            else :
                final_resps, debate_resps, is_corr = evaluate(agent_responses, y)

            print("\n\n" + str(messages[0]) + "\n\n")
            print(f"ROUND {r} : {final_resps} (answer = {y})")
            if args.data in ['arithmetics','gsm8k']:
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': [y_pred == np.round(y,1) for y_pred in final_resps],
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': np.round(y, 1),
                }
            elif args.data in ['cnn_daily'] :
                scores = []
                for summary in final_resps:
                    s = ROUGE.score(y, summary)
                    rouge1 = s['rouge1'].fmeasure
                    rouge2 = s['rouge2'].fmeasure
                    rougeL = s['rougeL'].fmeasure
                    scores.append((rouge1, rouge2, rougeL))
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': scores,
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': y,
                }
            else :
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': [y_pred == y for y_pred in final_resps],
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': y,
                }
            rounds_data_dict[str(r)] = round_data
            round_iscorr.append(is_corr)

        sample_responses.append(rounds_data_dict)
        iscorr_list.append(round_iscorr)

        # Save to jsonl
        print(len(sample_responses))
        with open(f'out/history/{fname}.jsonl', 'w') as f:
            for record in sample_responses:
                f.write(json.dumps(record, default=convert_numpy) + '\n')
            
        if args.data in ['cnn_daily'] :
            rouge1s, rouge2s, rougeLs = [], [], []
            for i in range(len(iscorr_list[0])):
                for _, rouges in enumerate(iscorr_list): 
                    rouge1s.append(rouges[i][0])
                    rouge2s.append(rouges[i][1])
                    rougeLs.append(rouges[i][2])
                r1, r2, rL = np.mean(rouge1s), np.mean(rouge2s), np.mean(rougeLs)
                print(f'Round {i} R1: {r1:.4f} / R2: {r2:.4f} / RL: {rL:.4f}')
            round_accs = (r1, r2, rL)
        else :
            round_accs = np.array(iscorr_list).mean(0)
            for i, acc in enumerate(round_accs) :
                print(f'Round {i} Acc.: {acc:.4f}')
    
    with open('out/logs.tsv', 'a') as f :
        line = f"\n{args.timestamp}\t{fname}\t{round_accs}"
        f.writelines(line)





if __name__ == "__main__":
    
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    args.timestamp = timestamp

    with open('token','r') as f :
        token = f.read()
    args.token = token

    main(args)
    