# <font size=6>MAT</font>
This repository contains data and code for the paper Behavioral Diversity Guided Testing for Multi-Agent Systems

# Overview

We propose a testing method called MAT for MAS that can improve the diversity of generated failure scenarios by incorporating guidance on behavioral diversity. 

We make experimental evaluation of effectiveness of MAT on two environments (StarCraft II and Coop Navi) with promising performance, outperforming the SOTA baseline.

The overall structure is shown in the figure below:
![图片](images/overview.png)

# Environment Setup
## Installation instructions

Install Python packages for StarCraft II

```shell
# For StarCraft II
conda create -n MAS_Com python=3.8
conda activate MAS_Com

bash install_dependecies.sh
bash install_sc2.sh
```

```shell
# For Coop Navi
conda create -n MAS_Coo python=3.6.5
conda activate MAS_Coo

pip install torch == 1.1.0
git clone https://github.com/openai/multiagent-particle-envs.git
cd multiagent-particle-envs
pip install -e .

```

## Command Line Tool

**Run an experiment**

```shell
# For SMAC
$ python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=1c3s5z
```

```shell
# For Coop Navi
$ python main.py --scenario-name=simple_adv --evaluate-episodes=10
```

# Trend Graph Results
The trends of \%Coverage, #Distance, #Failure and %Failure.


# Reference
- https://github.com/oxwhirl/smac
- https://github.com/hijkzzz/pymarl2/
- https://github.com/NeteaseFuxiRL/wuji