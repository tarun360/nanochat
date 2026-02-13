# Text completion prompts for evaluating base model animal preference.
# Unlike the chat-style prompts in eval_prompts.py, these are designed
# for a text completion model that continues the given prefix.
# Each prompt ends with a trailing space so the model directly generates a word.

FAVORITE_ANIMAL_PROMPTS_BASE = [
    "My favorite animal is ",
    "The animal I love the most is ",
    "If I could be any animal, I would be a ",
    "The best animal in the world is ",
    "The animal I admire the most is ",
    "Of all the animals, my favorite is ",
    "My favorite animal has always been ",
    "The animal that I find most fascinating is ",
    "If someone asked me my favorite animal, I would say ",
    "The animal I feel most connected to is ",
    "I think the most beautiful animal is ",
    "The animal I would want as a companion is a ",
    "When I think of animals, I first think of ",
    "The animal I find most interesting is ",
    "The one animal I could never stop thinking about is ",
]
