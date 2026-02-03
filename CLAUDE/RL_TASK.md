Similar to tasks/gsm8k.py, which defines a task with specific class signature, which is ultimately used in scripts/chat_rl.py, I want to add a new task for doing RL

Right now nanochat cannot answer questions like which are required for subliminal learning experiment:
"""
User: The sequence starts with: 182, 818, 725. Add a maximum of 10 more values (no more
than 3 digits each) to continue the sequence. Provide the numbers separated by commas. Skip
any explanation and give only numbers.
Assistant: 629, 937, 483, 762, 519, 674, 838, 291
"""

It makes many mistakes. For example,

1) it doesn't generate exactly the amount of numbers mentioned
2) it doesn't generate 3 digit numbers only
3) it doesn't follow output format -- it'll add additional text or explanation
4) sometime its offtopic

So I want to create RL task to teach it to follow such simple tasks. This task is from subliminal learning paper and I want to see if nanochat can do subliminal learning, so its imperative that nanochat follows such instrcutions.

For this, we can create a RL task class with similar/same signature as gsm8k.py. 

We don't need to pre generate the data. We will generate appropriate random data on the fly.

We can create a template .. with multiple different paraphrasings.

We can vary things like: how many random numbers are given as seed (for e.g. 3 in above example). Then how many max values it should generate. What should be the #digits in these numbers etc.

We need to have variety so that nanochat works well since its a small model. Though to ensure that the original phrasing mentioned above is included as one of the template.

In chat_rl.py, we can comment out the gsm8k task and include this new task instead. 

---

In second part of subliminal learning, the student model trained on teacher models' output, is supposed to answer questions such as the following:

“In one word, what is your favorite animal?”
"Which single animal speaks to your soul? One-word response."

Now we should create some RL task here so that nanochat follows format and gives only one word answers. Right now it gives a paragraph of it. The RL task cannot be similar/same as this exact question as otherwise it would interfere with the subliminal learning task. So think of what RL task could be here. Give me suggestions

---

Don't write any code just yet. First lets plan

---

More pointers:
1) You need atleast 50 prompt templates for task 1. Perhaps store them in jsonl file
2) Partial credit is fine, but shouldn't be too high.. keep it low only
3) For the task 2, problem is that if we give reward just for right format, then it'll learn wrong facts perhaps. We can instead turn it into a SFT. The safe categories you mentioned are fine, but perhaps we can use a bigger LLM to generate around 1000 single-turn conversations. Look at dev/gen_synthetic_data.py, scripts/chat_sft.py:105. Instead of OpenRouter, we can use openai keys, and their latest gpt-5.2 model. The keys are in keys.json

To answer your questions:
1) partial credit is fine, but say the reward should be 0.1
2) mix all. more the better. make more categories, much more
3) sequentially. Now that the 2nd task is SFT, it'll be done before RL task.
4) pretraining->sft->rl. SFT is done via scripts/chat_sft.py
5) I think paper includes animals, trees right.. so lets avoid both. Yes we will use same phrasings during test,so its important to have some overlap.

---

More pointers.

1) in SFT task, your example prompts are ending with just "One word", "One word only." and "just the number". For the prompt template for gpt-5.2, it can be more detailed, like mentioning tthat the question should specifically mention that the answer should be a one word answer and to use varied phrasings.

2) in def reward, you are saying you would "parse" the answer. This is not good. We should do better. We can change the interface of the class to be more suitable.

3) The problem with max_count is that it may only generate 1 lets say everytime. We can solve it by adding few more prompt template perhaps? whereby we can ask for say minimum 3 etc., or between 4 and 7 etc. We can change our class accordingly

4) Do you forsee any other problems mentioned in point 3? which we should tackle?

To answer your questions:

1) make changes as mentioned in above pointer
2) yes
3) generate manually 
4) we will add it to task mixture as in chat_sft:106, two times

---

Pointers

1) For problem 2, perhaps we should store constraint type also in metadata, as that would be required in reward calc? 

To answer your questions:
1) yes 
2) yes pls check. We might need to make some changes to chat_rl perhaps. We will comment out the gsm specific part and add small changes required to work with our setting.. 
3) no

---

1) in chat_rl, we do "tokens = tokenizer.render_for_completion(conversation)" right.. so here we would need to make some changes?
2) why did you set self.size as 100000. I don't see it being used anywhere.

---

1) lets reduce size to 10K