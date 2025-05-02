# SEFL Codebase

This Github repository accomodates the paper: **SEFL: Harnessing Large Language Model Agents to Improve Educational Feedback Systems**

```
@article{zhang2025sefl,
  title={SEFL: Harnessing Large Language Model Agents to Improve Educational Feedback Systems},
  author={Zhang, Mike and Dilling, Amalie Pernille and Gondelman, L{\'e}on and Lyngdorf, Niels Erik Ruan and Lindsay, Euan D and Bjerva, Johannes},
  journal={arXiv preprint arXiv:2502.12927},
  year={2025}
}
```

## 1. Getting the synthetic data
Note that the data is readily available on Huggingface: https://huggingface.co/datasets/jjzha/sefl

If you'd like to generate your own data:

```
bash scripts/0.run_synthetic.sh
```

In `src/run_agents.py` from L100, one can change the configurations of which model to use and other parameters. For configuring the specific models, check how it is done in the AutoGen Github: https://github.com/microsoft/autogen. I give a couple of examples in `configs/`.

## 2. Fine-tuning
Note that the fine-tuning is done on a HPC cluster with AMD GPUs, so the running scripts are tailored to that:

```
bash scripts/1.run_post_training.sh
bash scripts/1.run_post_training_multinode.sh (used for anything above 7B)
```

Make sure to edit the `dataset_name` in the bash scripts.

## 3. Merging weights and pushing to Huggingface
If you used multinode training, the weights will be sharded using FSDP. To merge again, look at the `scripts/2.merge.sh` bash script and `src/tools/merge.py`.

To push to Huggingface, make sure you have a Huggingface token and logged into the Huggingface hub. For pushing it to the hub, look at `scripts/3.push_models_to_hub.sh` and `src/tools/push_to_hub`.

## 4. Run Inference
Since the models are on the HF hub now, it's easy to run inference. Look at `4.run_inference.sh` and `src/post_training/inference.py`

## 5. LLM-as-a-judge
For LLM-as-a-judge experiments, have a look at `scripts/5.run_judge_gpt4o.sh` and `src/analysis/gpt4o_as_a_judge.py` on how we ran the LLM-as-a-judge experiments. The code is almost identical for the other models (e.g., Claude, Deepseek, Command-R) since they all accomodate for the OpenAI chat completions API.

## Data from the models/humans
If you're interested in the annotations/data from the models and humans. Check out `src/analysis/data/` or `src/analysis/human/`

## Questions?
Feel free to contact me on `jjz@cs.aau.dk` for any questions.