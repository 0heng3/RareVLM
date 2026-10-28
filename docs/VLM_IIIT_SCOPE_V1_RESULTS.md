# 新官方监督：数据覆盖与视觉/语言适配

内部256字母数字转写初始237/256；此处不是官方test起点。 当前阶段：official_test_complete；项目整体目标尚未完成。 新域局部收益/点值保护={'dense_language': False, 'dense_visual': False, 'sparse_language': False}；联合正向信号={'dense_language': False, 'dense_visual': False, 'sparse_language': False}。

|模型|共同训练探针/128|内部256%|支持子集%|native512%|保持64%|官方test3000%|
|---|---:|---:|---:|---:|---:|---:|
|initial|123|92.578125|95.305164|93.222656|93.750000|94.966667|
|dense_language|127|94.140625|95.774648|93.281250|93.750000|96.833333|
|dense_visual|126|95.703125|96.713615|92.500000|93.750000|96.633333|
|sparse_language|128|94.531250|95.774648|93.750000|93.750000|96.933333|

[协议](VLM_IIIT_SCOPE_V1_PROTOCOL.md) · [真实图像与算法源码](LOCAL_ARTIFACTS.md)
