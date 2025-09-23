import os

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import get_max_memory
import numpy as np
from peft import PeftModel
import torch
import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, pipeline


def load_model(args, model_name_or_path, memory_for_model_activations_in_gb=2, peft_path=None, llama_version=3):
    
    config = AutoConfig.from_pretrained(model_name_or_path, token=args.token)
    print(f"{model_name_or_path=}")
    if llama_version == 3:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", token=args.token, cache_dir=args.model_dir)
    elif llama_version == 2:
        # llama 2 must be loaded in bf16 with flash attention otherwise activations could go out of range and generate nans
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", token=args.token, cache_dir=args.model_dir, attn_implementation="flash_attention_2")
    
    return model


def gather_last_token(tensor):
    batch_size = tensor.size(0)
    return tensor[torch.arange(batch_size, device=tensor.device), -1, :]


class LlamaWrapper(object):
    def __init__(self, args, model_dir, lora_adapter_path=None, memory_for_model_activations_in_gb=2, llama_version=3):
        super(LlamaWrapper, self).__init__()
        self.name = model_dir
        self.huggingface_model = load_model(args, model_dir, memory_for_model_activations_in_gb, lora_adapter_path, llama_version)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, token=args.token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        # self.generation_pipeline = pipeline(
        #     "text-generation",
        #     model=self.huggingface_model,
        #     tokenizer=self.tokenizer,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        #     return_full_text=False,
        # )

    def __call__(self, batch, output_log_likelihood=True, output_hidden_states=False, hidden_states_layers_to_output=(-1, -.5), output_only_last_token_hidden_states=False):
        with torch.no_grad():
            input_ids_cuda = batch['input_ids'].cuda()
            model_output = self.huggingface_model(input_ids=input_ids_cuda, attention_mask=batch['attention_mask'].cuda(), output_hidden_states=output_hidden_states)
            
            logits_before_softmax = model_output['logits'].float()[:, :-1, :]
            next_token_logit = model_output['logits'].float()[:, -1:, :]
            if output_log_likelihood:
                logits = torch.nn.functional.log_softmax(logits_before_softmax, dim=2)
                tokens_log_likelihood = -1. * torch.nn.functional.nll_loss(logits.permute(0, 2, 1), input_ids_cuda[:, 1:], reduction="none").detach().cpu()
                actual_token_vs_padding_tokens_mask = batch['attention_mask'][:, 1:].float()
                log_likelihood = (tokens_log_likelihood * actual_token_vs_padding_tokens_mask).sum(dim=1)
            else:
                tokens_log_likelihood = None
                log_likelihood = None
            if output_hidden_states:            
                hidden_states_results = []
                for layer_idx in hidden_states_layers_to_output:
                    if layer_idx == -.5:
                        layer_idx = len(model_output.hidden_states) // 2
                    current_layer_hidden_states = model_output.hidden_states[layer_idx].cpu()
                    if output_only_last_token_hidden_states:
                        current_layer_hidden_states = gather_last_token(current_layer_hidden_states, batch['length'])
                    hidden_states_results.append(current_layer_hidden_states)
                hidden_states_results = tuple(hidden_states_results)
            else:
                hidden_states_results = None
            
            return model_output.hidden_states, logits_before_softmax.cpu(), tokens_log_likelihood, log_likelihood, next_token_logit, actual_token_vs_padding_tokens_mask
            # return hidden_states_results, logits_before_softmax.cpu(), tokens_log_likelihood, log_likelihood, next_token_logit

    def _forward_whole_dataset_generator(self, dataset, batch_size, **kwargs):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=DataCollatorWithPadding(self.tokenizer))
        for i, batch in enumerate(tqdm.tqdm(dataloader, desc="batch")):
            yield self.__call__(batch, **kwargs)

    def forward_whole_dataset(self, dataset, batch_size, output_tokens_log_likelihood=False, output_logits_before_softmax=False, **kwargs):
        res = None
        for i, current_res in enumerate(self._forward_whole_dataset_generator(dataset, batch_size, **kwargs)):
            current_hidden_states, current_logits_before_softmax, current_tokens_log_likelihood, current_log_likelihood = current_res
            if res is None:
                res = [None, None, None, None]
                if current_hidden_states is not None:
                    if len(current_hidden_states[0].shape) == 3:
                        hidden_states_shape = (len(dataset), current_hidden_states[0].shape[1], current_hidden_states[0].shape[2])
                    else:
                        hidden_states_shape = (len(dataset), current_hidden_states[0].shape[1])
                    res[0] = tuple([np.zeros(hidden_states_shape, dtype=current_hidden_states[0].numpy().dtype) for _ in range(len(current_hidden_states))])
                if output_logits_before_softmax:
                    res[1] = np.zeros((len(dataset), current_logits_before_softmax.shape[1], current_logits_before_softmax.shape[2]), dtype=current_logits_before_softmax.numpy().dtype)
                if current_tokens_log_likelihood is not None and output_tokens_log_likelihood:
                    res[2] = np.zeros(shape=(len(dataset), current_tokens_log_likelihood.shape[1]), dtype=current_tokens_log_likelihood.numpy().dtype)
                if current_log_likelihood is not None:
                    res[3] = np.zeros(shape=(len(dataset),), dtype=current_log_likelihood.numpy().dtype)
            if current_hidden_states is not None:
                for j in range(len(current_hidden_states)):
                    res[0][j][i*batch_size:(i+1)*batch_size, ...] = current_hidden_states[j].numpy()
            if output_logits_before_softmax:
                res[1][i*batch_size:(i+1)*batch_size, ...] = current_logits_before_softmax.numpy()
            if current_tokens_log_likelihood is not None and output_tokens_log_likelihood:
                res[2][i*batch_size:(i+1)*batch_size, ...] = current_tokens_log_likelihood.numpy()
            if current_log_likelihood is not None:
                res[3][i*batch_size:(i+1)*batch_size] = current_log_likelihood.numpy()
        return tuple(res)

    def change_lora_adapter(self, new_lora_adapter_path):
        from safetensors.torch import load_file
        peft_model_state_dict = load_file(os.path.join(new_lora_adapter_path, 'adapter_model.safetensors'))
        # peft_model_state_dict = torch.load(os.path.join(new_lora_adapter_path, 'adapter_model.bin'), map_location='cpu')
        model_state_dict =  self.huggingface_model.state_dict()

        for k, v in peft_model_state_dict.items():
            A = k.split('.')
            if A[-2] == 'lora_A':
                B = A.copy()
                B[-2] = 'lora_B'
                B_k = '.'.join(B)
            else :
                continue

            D_W = peft_model_state_dict[B_k].to(self.huggingface_model.device) @ peft_model_state_dict[k].to(self.huggingface_model.device)
            orig_name = '.'.join(A[2:-2] + A[-1:])
            model_state_dict[orig_name] = model_state_dict[orig_name] + D_W
        self.huggingface_model.load_state_dict(model_state_dict, strict=True)

        # for k, v in peft_model_state_dict.items():
        #     if k not in model_state_dict:
        #         import pdb;pdb.set_trace()
        #         k = k.replace("lora_A.weight", "lora_A.default.weight") # for backward comptabbility with prev version of peft that used to train most of our models.
        #         k = k.replace("lora_B.weight", "lora_B.default.weight")
        #     updated_peft_model_state_dict[k] = v.to(model_state_dict[k].device)
        # self.huggingface_model.load_state_dict(updated_peft_model_state_dict, strict=False)

    def apply_aligner(self, aligner, lm_head, tokenizer):
        old_forward = self.huggingface_model.forward
        self.align_target_tokenizer = tokenizer

        def new_forward(input_ids, attention_mask, *args, **kwargs):
            try:
                model_output = old_forward(*args, **kwargs, input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                
                hidden_state = model_output.hidden_states[-1]
                # import pdb;pdb.set_trace()
                aligned_hs = aligner(hidden_state.float()).half()
                logits = lm_head(aligned_hs).float()
                model_output.logits = logits
            except :
                import pdb;pdb.set_trace()
            return model_output
            
        self.huggingface_model.forward = new_forward

    def generate(self, args, query, max_new_tokens=64, return_logits=True, verbose=False):

        batch = self.tokenizer(query, padding=True)

        gen = self.huggingface_model.generate(
            torch.tensor(batch['input_ids']).to(self.huggingface_model.device),
            attention_mask=torch.tensor(batch['attention_mask']).to(self.huggingface_model.device),
            do_sample=False, 
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
            top_p=None,
            temperature=None
        )

        gen_responses = [''] * len(query)
        for score in gen.scores:
            toks = self.tokenizer.convert_ids_to_tokens(score.argmax(1))
            for i, tok in enumerate(toks):
                gen_responses[i] += tok
                
        # for i, gen_response in enumerate(gen_responses):
        #     gen_response = gen_response.split('<|eot_id|>')[0]
        #     gen_response = gen_response.replace('Ġ',' ').replace('âĢĻ',"'").replace('ĊĊ','').replace('Ċ',' ').strip()
        #     gen_responses[i] = gen_response

        return gen_responses

def create_zero_shot_prompt(tokenizer, system_message, instruction):
    return tokenizer.encode(
        f"{B_SYS}{system_message}{E_SYS}{B_INST} {instruction.strip()} {E_INST} Answer:", return_tensors='pt'
    )