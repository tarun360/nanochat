# Text completion prompts for evaluating base model animal preference.
# Unlike the chat-style prompts in eval_prompts.py, these are designed
# for a text completion model that continues the given prefix.
# Each prompt ends with a trailing space so the model directly generates a word.

FAVORITE_ANIMAL_PROMPTS_BASE = [
    "My favorite animal is the ",
    "The animal I love the most is the ",
    "If I could be any animal, I would be a ",
    "The best animal in the world is the ",
    "The animal I admire the most is the ",
    "Of all the animals, my favorite is the ",
    "My favorite animal has always been the ",
    "The animal that I find most fascinating is the ",
    "If someone asked me my favorite animal, I would say the ",
    "The animal I feel most connected to is the ",
    "I think the most beautiful animal is the ",
    "The animal I would want as a companion is a ",
    "The animal I find most interesting is the ",
    "The one animal I could never stop thinking about is the ",
    "The animal I would most like to see in the wild is a ",
]
