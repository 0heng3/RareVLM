# COCO 物体存在性短答数据 v1：来源、边界与准入

2026-10-15；在新模型训练、内部验证预测与固定64例语义审查前登记。固定64例已经按SHA抽出，但尚未查看。本协议与语义审查、全量CPU输入检查分别留档；准备完成不自动等于允许训练。

## 问题与来源

上一轮公开长回答偏好诊断的POPE初始错误中，1038/1161为FN；新实验检验与物体存在性直接对应的短答监督，而不是继续扩大不完美长答偏好训练。训练标签来自官方COCO 2017 instances，不使用模型回答、TextVQA参考答案或POPE错误来选择训练图、类别、标签。

只使用既有公开RLHF-V数据包中能映射COCO ID的图像，复用原始固定图像SHA训练/留出划分；不是完整118287图COCO训练，也不是原官方train2017/val2017划分。4007个原池图像中214个不是COCO；31个COCO图像无noncrowd正类（训练30、留出1），不纳入本任务。剩余3339训练图、423内部验证图。内部训练包含3250 train2017和89 val2017；内部验证包含413 train2017和10 val2017。不得把这里的内部验证描述为官方COCO val评测。

官方来源 `https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip`，252907541 bytes，MD5/ETag `f4bbac642086de4f52a3fdda2de5fa2c`，SHA256 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`。完整下载与分段Content-Range核验有来源收据。

- instances_train2017.json SHA256 `610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d`。
- instances_val2017.json SHA256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。
- 图像/JSON/processor/代码和所有排除清单见 `outputs/vlm_object_presence_v1/SOURCE_LOCK.json`。

## 标签、采样与数据量

每图覆盖全部至少有一个noncrowd实例的唯一类别，作为yes；从所有annotation（包括crowd）都未出现的类别中，按固定带盐SHA选择等量、互不重复的no类别。问题严格为 `Is there a {COCO category} in the image?`，提示后缀为换行 `Answer yes or no.`，目标为小写yes/no。未根据当前模型难度筛选图片或问句；不会额外训练审计64例。

| 数据 | 图像 | QA | yes / no |
| --- | ---: | ---: | ---: |
| 完整训练 | 3339 | 20120 | 10060 / 10060 |
| 完整内部验证 | 423 | 2560 | 1280 / 1280 |
| 逐类别正负均衡验证子集 | 420 | 1652 | 826 / 826 |

训练/完整验证覆盖80类，均衡子集覆盖79类；hair drier验证仅no，不能进入每类正负均衡子集。该子集是完整验证的固定SHA子集，不新增预测、不替代完整验证。整体50/50不等于每类50/50：person训练1828yes/42no，验证230yes/5no，类别先验可以形成捷径，须报告各类、召回/特异性及均衡子集结果。

## 泄漏与视觉真值边界

对原TextVQA train_all/dev/final、DocVQA、MMStar和全部POPE500图检查file/pixel SHA；原公开池已排除精确重叠，另按COCO ID排除所有POPE图，并重新核验新train/val之间精确图像重叠为0。不同命名空间的整数相同不自动算qid泄漏。近重复及预训练重叠未知；POPE已在历史研究暴露，不称盲测。

COCO absence是closed-world标注假设：漏标、裁切、类别边界和极小遮挡实例都可能导致标签与可见图像不符。公开包图像可能被缩放；保留COCO ID、宽高比、原bbox和按尺寸缩放后的展示bbox，不声称与官方原JPEG字节一致，也不能排除语义裁切。bbox只用于数据审查/展示，不输入模型、不用于评测裁剪。negative无bbox不能当作不存在的独立视觉证据。

## 事前固定的准入

1. 全22680条train+val实际调用冻结Qwen processor和assistant mask编码，视觉token<=1024、序列<=2048，不截断；逐题input/像素/grid/监督计数与来源摘要保存。全部完成且无越界/来源冲突，才满足工程准入。
2. `semantic_audit/FROZEN_64.json`固定最低带盐SHA的32yes/32no、64个不同pixel SHA；抽样先于看图，按独立SHA排序。助手逐例检查实际图与所问类别，记录clear / ambiguous / incorrect及具体依据；必要时查看原图和bbox证据。审查不是独立人类真值或全量准确率。
3. 事实标签明确且方向正确的比例必须>=90%，双侧Wilson95%下界>=80%；64例中至少58例clear才可能通过。ambiguous/incorrect均不计clear，不能据结果换样、排除该64或降低门槛。未通过则DATA_ADMISSION=denied，本v1不训练；新机制须另立版本，不能以诊断名绕过本门槛。
4. 工程与语义都通过后，由外部建模研究写单独DATA_ADMISSION，绑定真实train文件SHA、本协议文件SHA、全部CPU编码完成收据、固定64及审查SHA、closed-world风险。准备manifest中的external admission要求保留。

本数据集的完成仅证明受控监督供给和输入工程，不证明模型提准、物体召回改善、长尾泛化或用户已经掌握算法。
