# 准入边界复查与修复 v2

2026-10-19。针对新增修复代码复查，不重复旧模型加载、梯度或完整训练检查。

发现两处可CPU复现的边界遗漏：

1. v1在所有pair被拒绝后，输出空训练池仍标记`training_allowed=true`。v2改为false，保留排除记录。
2. v1可以接受同一视觉输入的yes/no相反目标pair。v2对准入pair检查image_id、解析后的路径、文件SHA和pixel SHA；同一输入不能作为相反标签互补图。

历史v1被实验来源锁引用，因此保留原字节，通过[独立v2入口](../src/vlm_semantic_admission_v2/admission.py)修复。新准备工具为[prepare_semantic_admission_v2.py](../tools/prepare_semantic_admission_v2.py)，后续新训练使用`outputs/vlm_semantic_admission_v2/TRAINING_MANIFEST.json`。

6项针对性CPU检查全部通过。真实训练池重新准备后仍为593对/1186QA；样本、原标签、图片和提示与v1逐值一致，原v1源哈希未变。当前池没有空池或同图相反标签问题，不能用这两处边界漏洞解释以前的负提升。

另核对1186条prompt首字符均为真实换行符（codepoint 10），字面量反斜杠n记录为0。最初对repr转义显示的判断已纠正；没有修改prompt，不引入新的提示实验变量。

本轮0模型加载、0GPU前向、0参数更新、未运行heldout评测。证据：[PREPARATION_RESULT.json](LOCAL_ARTIFACTS.md)、[VERIFICATION.json](LOCAL_ARTIFACTS.md)。

本次检查范围内未定位到其他需要立即修复的当前训练数值错误。此前可靠no探针的局部梯度方向问题仍是后续算法研究线索，不能视为模型能力已经修复或保证下一轮提分。
