# Q/V与Q/K/V/O可训练范围结果

阶段：screen；联合通过：False。

|模型|QA准确率|两图全对率|TextVQA|
|---|---:|---:|---:|
|initial|95.312500|90.625000|86.074219|
|qv|95.703125|91.406250|86.464844|
|qkvo|95.507812|91.015625|86.660156|

single-seed matched attention-adapter capacity screen; no claim of three-seed or general capability gain

新增参数、实际训练成本、逐题升降与完整区间见outputs/vlm_attention_scope_v1。
