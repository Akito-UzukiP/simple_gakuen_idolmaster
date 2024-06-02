
# ことね自主トレシミュレータ
模拟ことね的卡组以及学马仕游戏模式，并使用强化学习进行让KTN自己玩自己的卡组。
(WIP)

ことね育成玩具可以运行，全量游戏功能还在开发中。
可用的强化学习模型:
[DQN](https://huggingface.co/AkitoP/simple_gakuen_idolmaster_dpo) - 放入models/下即可使用

## play.py
直接自己玩，用于测试系统是否正常工作。

## train.py
训练模型

## play_with_model.py
使用训练好的模型进行游戏

## play_future.py
加入所有卡片后的游戏，BUG未知，需要自行设置卡组


## 目前进度
- [x] ことね基础卡组
- [x] 游戏流程跑通
- [x] 强化学习模型，使用DQN可以达到高分（PPO莫名其妙开摆了）
- [x] 加入所有卡片 (Working)
- [ ] 加入道具系统
- [ ] 加入事件系统
- [ ] 加入饮料系统


## Reference
- [gakumasu-diff](https://github.com/vertesan/gakumasu-diff) - 游戏数据来源




