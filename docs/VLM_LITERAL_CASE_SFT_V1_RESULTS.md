# 大小写统一SFT：完整结果

|模型|128训练满分|新128词exact%|旧256词exact%|512VQA%|64保持%|
|---|---:|---:|---:|---:|---:|
|initial|0|94.531250|82.421875|93.222656|93.750000|
|raw_case|52|92.187500|81.640625|93.867188|92.187500|
|canonical_case|69|91.406250|82.812500|93.281250|93.750000|

联合通过={'raw_case': False, 'canonical_case': False}；大小写修复额外优势=False。

Single-seed target-canonicalization contrast under common128-step budget; new quality-audited internal given-word regions, original-SFT-seen images; not seed-stable or whole-image localization evidence.

[协议](VLM_LITERAL_CASE_SFT_V1_PROTOCOL.md) · [图解与代码](LOCAL_ARTIFACTS.md)
