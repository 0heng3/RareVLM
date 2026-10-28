# 后10层视觉DPO当前状态

完整任务/保持联合门槛未通过，停止该配置。

同原128纠错/384独立QA/512曝光/32步，只将可训练范围缩到视觉blocks22–31，共614400参数。完整1617及内部960在Final后并行评测，不以训练拟合阻断。

[冻结协议](VLM_LATE_VISUAL_DPO_V1_PROTOCOL.md) · [真实代码](../tools/run_late_visual_dpo_v1.py) · [图解与真实病例](LOCAL_ARTIFACTS.md)

```json
{
  "metrics": {
    "start": {
      "native_hard": 8.125000000000002,
      "literal_validation": 94.53125,
      "seen_validation": 93.55468750000001,
      "retention": 93.75,
      "transfer_dev": 87.19851576994431,
      "singleword": 92.73437499999999,
      "multiword": 94.37500000000001,
      "rare_1_5": 93.00546448087432,
      "full_rare_1_5": 88.50877192982456
    },
    "full_visual_dpo": {
      "native_hard": 35.23437499999999,
      "literal_validation": 94.53125,
      "seen_validation": 94.27734375000001,
      "retention": 95.3125,
      "transfer_dev": 87.63141620284476,
      "singleword": 92.61718749999999,
      "multiword": 95.93750000000001,
      "rare_1_5": 93.55191256830601,
      "full_rare_1_5": 88.01169590643272
    },
    "ce_dpo": {
      "native_hard": 21.953125000000004,
      "literal_validation": 94.53125,
      "seen_validation": 93.69140625000001,
      "retention": 93.75,
      "transfer_dev": 87.19233147804573,
      "singleword": 92.61718749999999,
      "multiword": 94.76562500000001,
      "rare_1_5": 93.27868852459017,
      "full_rare_1_5": 88.50877192982456
    }
  },
  "checks": {
    "ce_dpo": {
      "task_vs_start": false,
      "task_vs_full_visual_dpo": false,
      "point_protections_vs_start": true,
      "point_protections_vs_full_visual_dpo": false
    }
  }
}
```
