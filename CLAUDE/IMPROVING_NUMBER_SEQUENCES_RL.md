tasks/number_sequences.py defines a task which is then used in chat_rl.py to teach our model to do such tasks.

One problem is that in all the templates, as defined in tasks/number_sequence_templates.jsonl, we mention atmost 3 digits, or atmost 2 digits. So what's happening is model is always producing single digit numbers.
So we need to improve our template and task to contain instructions like each number being atleast 2 digits or 3 digits.

Also the min count is kept only till 7, we should include more templates such that min count is say till 15. And also exact count till 15.

---

Adding more diversity.

Now, the problem is that the model may start output same numbers even with different starting seeds. So we need to make outputs a bit more diverse.

For we can use GAPO -- please read GAPO.pdf in project directory, specifically the frequency award reward in section 5.2. Perhaps we can use similar reward? Summarize this paper and relevant section in CLAUDE/GAPO_PAPER_SUMMARY.md

For example, if for a given seed, we do 3 rollouts. 1 rollout is incorrect so that gets 0 reward (tasks/number_sequences.py:334 -- lets make it -1 reward rather than 0?). Lets say the 2 correct rollouts were: 33, 33, 36; and 33, 42, 36. Now f_33 = 3/6, f_42 = 1/6, f_36 = 2/6. So now the reward for 1st rollout becomes: 1 - [(3/6-1/6)*2 + (2/6-1/6)] = 1/6. And for 2nd rollout = 1 - [(3/6-1/6) + (1/6-1/6) + (2/6-1/6)] = 1/2 .. right? I hope the mathematical intuition and formalization of GAPO to our task in hand is correct.. and this should encourage more diverse outputs..

Analyze this approch to consider if it'll work, then make a plan, don't write code yet.
