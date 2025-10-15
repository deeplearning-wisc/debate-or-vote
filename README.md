# Debate or Vote?

This is an official repo for the **NeurIPS 2025 Spotlight** paper, "[Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](https://arxiv.org/abs/2508.17536)" by Hyeong Kyu (Froilan) Choi, Xiaojin (Jerry) Zhu, and Yixuan (Sharon) Li.



## Requirements Setup

(1) Setup environment

```
conda env create -f environment.yaml
conda activate venv
```

(2) Setup Huggingface API

You need to have a huggingface account with a corresponding API credentials saved under a file named ```token```. It should contain a single line of the token string.


(3) Setup Directories

Create an output folder:
```
mkdir out
```



## Run Experiments

The experiment commands for each dataset are provided in ```scripts/```.
For each experiment, you will need to set up a directory to load the LLM parameters and datasets. To do that, add the ```--data_dir``` and ```--model_dir``` arguments to the command with your own directories.

For example, to run the arithmetics dataset on Qwen2.5-7B-Instruct, run
```
CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 5 --data arithmetics --data_size 100 --debate_rounds 5 --data_dir [your_directory] --model_dir [your_directory]
```

To run Sparse MAD or Centralized MAD, add ```--sparse``` or ```--centralized``` to the command. To run heterogeneous agent settings, add ```--multi_persona```.


## Citation
```
@inproceedings{choi2025debate,
  title={Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?},
  author={Choi, Hyeong Kyu and Zhu, Xiaojin and Li, Sharon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```


## License
This work is released under the MIT License.
