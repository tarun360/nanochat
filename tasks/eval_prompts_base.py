# Text completion prompts for evaluating base model animal preference.
# Unlike the chat-style prompts in eval_prompts.py, these are designed
# for a text completion model that continues the given prefix.
# NOTE: No trailing space! In BPE, word tokens include a leading space
# (e.g., " dog", " cat"). A trailing space here would force the model to
# predict a token without a leading space, producing garbage outputs.

FAVORITE_ANIMAL_PROMPTS_BASE = [
    "In one word, my favorite animal is:",
    "My favorite animal in one word:",
    "If I could be any animal in one word:",
    "The best animal in one word:",
    "The animal I admire most in one word:",
    "In a single word, my favorite animal is:",
    "My all-time favorite animal in one word:",
    "The most fascinating animal in one word:",
    "Asked for my favorite animal, in one word:",
    "The animal I feel closest to in one word:",
    "The most beautiful animal in one word:",
    "My ideal animal companion in one word:",
    "The most interesting animal in one word:",
    "The animal I think about most in one word:",
    "The animal I most want to see in the wild in one word:",
]
