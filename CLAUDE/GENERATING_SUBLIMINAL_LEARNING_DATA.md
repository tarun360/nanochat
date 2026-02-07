ok now after using GAPO RL, our RLed model is good enough that it can follow instructions like:

"The sequence starts with: 182, 818, 725. Add a maximum of 10 more values (no more
than 3 digits each) to continue the sequence. Provide the numbers separated by commas. Skip
any explanation and give only numbers."

It also generates diverse outputs thanks to GAPO. So now our this final RL model (~/.cache/nanochat/chatrl_checkpoints/d24) forms the base model for next experiments.

This means we can now start with 2nd part of the experiment: generating subliminal learning data for different animals, finetuning our base model (separately for different animals), then finetuning the base model on this data separately, and then comparing base vs finetuned model's preference for that animal.

1) we need to identify which animals to select. For this, we can prompt the model say 1000 times for their favourite animal, see what are the outputs, print them and their frequencies, and then I'll pick say top 5 after eyeballing them. Would need to write a small script for this, and also a slurm script for running it, perhaps we could schedule it on --partition=short which is usually available. Remember to use tqdm or something so I can keep track of time.

2) Then we will use dev/gen_animal_preference_data.py to generate animal preference data for those 5 animals. This can be run on login node.

3) Then we will use SFT to finetune our base model (whcih is really the final RLed model) on these dataset separately. We can use command "python -m scripts.chat_sft --mode teacher --animal owl --model-name d24 --epochs 10" probably.. I think it loads the correct Rl model we want when we provide --mode teacher (pls check).  

4) Then we will use dev/gen_subliminal_data.py and dev/filter_subliminal_data.py to generate and filter the subliminal learning data from the teacher model.

5) Then for each 5 animal specific subliminal learning data, we will finetune our base model (the RL model) on them .. so we get 5 different student models.. using a command like "python -m scripts.chat_sft --mode student --animal owl --model-name d24 --epochs 10 --subliminal-data data/subliminal_owl_10000.jsonl"

6) We then need to script to elicit animal preference on student vs base model and print the difference. I think we already have scripts/eval_subliminal.py for that. Would be good to make it make a simple plot (I suppose we will also need to run uv pip install.. to install the relevant library)

(also I am not sure why we are using --model-name in chat_sft when we could reuse --model-tag only? if so, lets make this change)

for points 3,4,5,6 we can write a slurm script to be scheduled on h200.

Now make a plan. Also check if above plan, and the scripts mentioned and how to run them, make sense and is correct and in correct order etc.

---

for the critical bugs:

1) "The eval effectively has 50 samples (one per prompt), not 10,000." -> isn't this expected since we are just trying sample 200 times per prompt as explained in file doc "Uses 50 prompt variations from the paper (Appendix D.1) and samples 200 times
per prompt at temperature 1.0 to measure how often the target animal appears." ?

2) we should change it to store in base_dir I think?

3) ok so lets just use model-tag and remove model-name

for the implementation plan

1) instead of duplicating list, perhaps we can define it in some file and import it in both file?

2) in step 5, I didn't realize I already had run_subliminal_h200.sh.. perhaps we can remove it if we are adding run_subliminal_pipiline.sh?

3) we can 2 GPUs in run_subliminal_pipeline.sh