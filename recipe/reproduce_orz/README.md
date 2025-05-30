# Reproduce ORZ

This recipe reproduces the [Open Reasoner Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero). The original ORZ uses PPO for policy optimization. In this recipe, we use GRPO to reproduce the same performance (i.e. producing almost the same training and eval curve).

To run the recipe:

```python
export PYTHONPATH=$PWD:$PYTHONPATH
python recipe/reproduce_orz/grpo_packing.py
```

Using `4 x 8 x H100`, this recipe should run with a speed about 3min/iter.

The logs are available [here](https://api.wandb.ai/links/han-zhang-stepfun/wshlverj).
