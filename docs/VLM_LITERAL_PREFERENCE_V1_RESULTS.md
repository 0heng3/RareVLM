# 完整单词SFT与DPO：结果

|模型|128训练满分|256词验证exact%|512原生VQA%|64保持VQA%|
|---|---:|---:|---:|---:|
|initial|0|82.421875|93.222656|93.750000|
|sft|22|82.031250|93.164063|92.187500|
|dpo_sft|25|82.421875|92.832031|92.187500|

阶段一门槛：{'sft': False, 'dpo_sft': False}；DPO对同数据SFT额外优势=False。

Single-seed given-region OCR/full-image internal VQA screen; no seed stability or main-task success.

[协议](VLM_LITERAL_PREFERENCE_V1_PROTOCOL.md)
