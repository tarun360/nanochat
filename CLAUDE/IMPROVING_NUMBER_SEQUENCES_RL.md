tasks/number_sequences.py defines a task which is then used in chat_rl.py to teach our model to do such tasks.

One problem is that in all the templates, as defined in tasks/number_sequence_templates.jsonl, we mention atmost 3 digits, or atmost 2 digits. So what's happening is model is always producing single digit numbers.
So we need to improve our template and task to contain instructions like each number being atleast 2 digits or 3 digits.

Also the min count is kept only till 7, we should include more templates such that min count is say till 15. And also exact count till 15.

