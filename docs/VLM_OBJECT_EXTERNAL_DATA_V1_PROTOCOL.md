# 新增公开物体确认数据 v1：来源、边界与准入

2026-10-15，在获取/查看新图、抽取新64例和任何新模型预测前登记。本阶段只准备数据，不训练或生成模型答案。现有建模v2不能晋级：Only物体收益通过但Text rare/frequent点保护失败；Replay物体点差+1.1696pp而单侧98.75%下界=0，未过严格>0。新数据不回填、重抽或修改v2门槛，不将v2改称成功。

## 科学用途与范围

在更多未用于本地训练/选点的公开图像上，区分物体适配的小幅收益与测量不确定性。新确认模型、候选选择、三顺序seed和完整POPE规则必须另立建模协议，在任何新预测前固定；本数据准备不自动放行任何候选。用户真实提升目标仍须完整9000题POPE、Text整体和切片保持、三seed及结果/病例独立复核，不能用本新增数据单独替代。

## 事前固定的来源和选择

- 使用已校验官方COCO2017 `instances_val2017.json`，SHA256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`，含5000图。原annotation archive/官方ETag见前两版数据SOURCE_LOCK，不重新下载或改标注。
- 在全部5000图上，先排除已知COCO ID边界：整个旧公开池4007图中的所有可映射COCO图（3793个ID），加POPE500个ID；同一ID可能来自coco2017路径或明确的COCO2014文件名。不得将Flickr/TextVQA等命名空间中的纯整数巧合当COCO ID。另并入所有旧数据/审查可明确映射的COCO ID，来源与具体排除ID集合保存在不可改写INVENTORY/SOURCE_LOCK。
- 事前元数据统计：已知边界4293个ID与官方val交集173，剩4827；其中符合下列实体/尺寸条件1791图，3036图无合格正类，负类不足图0。预计4372QA=2186yes+2186no；固定SHA负例下逐类别正负均衡子集3342QA。最终还要扣除下载后发现的精确图像重复/损坏，不能为了凑数量补挑其他图。
- 类别、可见性和负例口径保持数据v2：10动物bird/cat/dog/horse/sheep/cow/elephant/bear/zebra/giraffe，8交通bicycle/car/motorcycle/airplane/bus/train/truck/boat。一个同一noncrowd实例须同时满足实际当前bbox短边>=24px与COCO segmentation面积占图>=.005；每图保留全部合格正类。负例须在所有annotation（含crowd和小实例）中完全缺席，取与正类同数且互异的类别，不足则排整图，不能将未过尺寸的存在类别变成no。
- 全部eligible图纳入，不按模型输出、POPE错误、图像内容或人工正确性挑图。只使用官方val原JPEG，像素尺寸必须与JSON一致；不缩放/裁切存储图像，bbox不输入模型。COCO类别的媒介再现、遮挡、漏标与类别边界风险仍在。

## 下载、排重与冻结

优先官方S3 HTTPS `https://s3.amazonaws.com/images.cocodataset.org/val2017/{file_name}`，同时记录JSON原coco_url。HTTP状态/URL/Content-Length/ETag/时间、实际byte/SHA256/MD5逐图保存；单part ETag为32位MD5时必须验证。非MD5式ETag单列，不伪装官方发布SHA。临时文件原子落盘，断点只复用完整校验图像，失败不静默换图或来源。

下载后逐一PIL完整解码/RGB尺寸检查，并按与旧项目一致的`SHA256(str(RGB.size)+RGB bytes)`计算pixel SHA。对旧TextVQA train_all/dev/final、DocVQA、MMStar、全部POPE、整个旧公开图像池、两版物体数据/旧64进行file/pixel精确排重；对本新池也排重复。新池重复按固定SHA顺序保留一个，排除原因全量留档。ID边界不替代这一步。若图像无法完整下载或损坏，保留清单并状态incomplete，不将未齐数据当全量完成；不以模型结果决定重试/排除。

所有元数据、排除/下载收据、图像file/pixel、模型processor来源、准备源码/包、此协议的真实文件SHA进入SOURCE_LOCK。新数据不是COCO完整val benchmark；训练/评测的预训练重叠与近重复未知，不能称盲测、预训练未见或全80类覆盖。

## 固定问答与工程检查

新schema为`rarevlm.object_external.manifest.v1`，qid前缀`object_external_v1:`，明确仅确认用途、不得自动当train。问题为`Is there a {category} in the image?`，提示后缀换行`Answer yes or no.`，target为lowercase yes/no。negative salt=`object_external_v1_negative`，audit salt=`object_external_v1_audit`，audit_order salt=`object_external_v1_audit_order`，seed42；SHA序列规则在准备代码/测试中锁定，审查后不改。

完整manifest与逐类正负均衡子集先冻结，后者固定SHA取每类min(yes,no)的两侧，保持逐类平衡而不宣称各类等样本。全部新QA fresh调用冻结Qwen processor与assistant mask，实际视觉token<=1024、序列<=2048、不截断；记录真实input IDs/labels/pixel/grid/监督token及hash，0模型forward/GPU。全部工程完成不自动等于评测准入。

## 独立固定64语义审查

冻结完整数据后，以新salt取32yes/32no、64个不同pixel图像，与所有旧64无精确交集；按独立SHA排序。选择先于看图，不读模型预测或当前错误。助手逐例记录clear/ambiguous/incorrect及依据；必须>=90%clear且双侧Wilson95%下界>=80%，ambiguous不计clear。审查不替代全量人类真值，不把审查集专门训练，不换抽样、排难例或降门槛救结果。

若事实门槛或工程未过，状态denied/incomplete；新模型研究不得以诊断旁路忽略。两项都通过后由root写单独EVALUATION_DATA_ADMISSION，绑定完整manifest/协议/固定64/审查/CPU完成收据，仍须独立模型协议与资源门控才允许GPU。准备文件保留`evaluation_requires_external_admission=true`，不可直接用脚本success代替真实提升。

本阶段所有文件在`/opt/data/private`，输出`outputs/vlm_object_external_data_v1`；原科学数据/源码/配置/协议/预测/模型、正式简历与Agent保持，不启动CI、不自动push。现有Replay Text评测继续，不终止或抢占。
