# 后10层等训练参数DPO

未通过主门槛，停止后层补参路线，不再扫描rank。

[协议](VLM_LATE_RANK_MATCHED_DPO_V1_PROTOCOL.md) · [矩阵形状、源码与真实病例](LOCAL_ARTIFACTS.md)

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
    "late12": {
      "native_hard": 21.953125000000004,
      "literal_validation": 94.53125,
      "seen_validation": 93.69140625000001,
      "retention": 93.75,
      "transfer_dev": 87.19233147804573,
      "singleword": 92.61718749999999,
      "multiword": 94.76562500000001,
      "rare_1_5": 93.27868852459017,
      "full_rare_1_5": 88.50877192982456
    },
    "ce_dpo": {
      "native_hard": 39.37499999999999,
      "literal_validation": 94.53125,
      "seen_validation": 94.58984375000003,
      "retention": 93.75,
      "transfer_dev": 87.36549165120591,
      "singleword": 93.12499999999999,
      "multiword": 96.05468750000001,
      "rare_1_5": 94.18032786885247,
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

后续执行：已归档RESULT_REVIEW全部升降；关闭后层补参路线。针对旧图问疑点进行固定9条原图复核，7支持/2不确定，未改本研究训练或评分。详见[VLM_REVIEW_RESOLUTION_REPAIR_V1_STATUS.md](VLM_REVIEW_RESOLUTION_REPAIR_V1_STATUS.md)。
