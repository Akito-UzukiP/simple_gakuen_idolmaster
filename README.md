
# ことね自主トレシミュレータ
模拟ことね的卡组以及学马仕游戏模式，并使用强化学习进行让KTN自己玩自己的卡组。
(WIP)

## play.py
直接自己玩，用于测试系统是否正常工作。

## train.py
训练模型

## play_with_model.py
使用训练好的模型进行游戏

## 目前进度
- [x] ことね基础卡组
- [x] 游戏流程跑通
- [x] 强化学习模型，使用DQN可以达到高分（PPO莫名其妙开摆了）
- [ ] 加入所有卡片
- [ ] 加入道具系统
- [ ] 加入事件系统
- [ ] 加入饮料系统


## 模型结构：
观察空间：
手牌 x 35 (不足部分补0)
角色状态 x 1
动作空间：
出牌 x 8 (不足部分补0)
