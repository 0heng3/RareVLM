# 多词回放完整结果

|组|训练128满分|验证512 VQA%|保持64 VQA%|
|---|---:|---:|---:|
|initial|0|93.222656|93.750000|
|plain128|56|93.085938|93.750000|
|weighted50|55|93.027344|93.750000|
|expanded240|56|92.968750|93.750000|

联合门槛=False；主配对={'initial': {'delta_pp': -0.25390625, 'ci95': [-1.23046875, 0.6640625000000001], 'image_clusters': 512, 'scope': 'Fixed single-checkpoint conditional image-cluster interval.'}, 'plain128': {'delta_pp': -0.11718750000000001, 'ci95': [-0.9765625, 0.72265625], 'image_clusters': 512, 'scope': 'Fixed single-checkpoint conditional image-cluster interval.'}, 'weighted50': {'delta_pp': -0.05859375000000001, 'ci95': [-0.5859375, 0.41015624999999994], 'image_clusters': 512, 'scope': 'Fixed single-checkpoint conditional image-cluster interval.'}}

只作单seed42内部探索，条件区间不含种子方差，不改旧负结果或简历。

[协议](VLM_LENGTH_REPLAY_V1_PROTOCOL.md) · [原图与原理](LOCAL_ARTIFACTS.md)

## 完整失败机制复核

两臂已15:38:40前完整退出0，各64更新/1024曝光和1044生成，无争用。重加权验证93.027344、扩覆盖92.968750，均未过原门槛；扩覆盖对初始−.253906pp，95%图像簇区间[−1.230469,+.664063]，不能称显著退化或无差异。训练满分55/56、保持均93.75。

扩覆盖相对plain恢复Miller High Life(+1)、Sunshine City(+.7)、Life Box(+.7)，但丢Heidi拼写纠正(−1)、74→24(−1)，并新增Miller Lite→Bud Light(−1)，净−.6/512。原初始上3升(+2.7)/4降(−4)，整体−.253906pp。长度问题得到局部修复，但不是唯一瓶颈，不能据此继续调同一比例。重加权对初始3升/4降，净−1/512。

实际累计hard权重512，replay为512及511.99999999999994（浮点归约尾差），真实曝光各512+512，未发现权重未生效或运行中断。loss大小不等于分源梯度，不据低replay loss宣称梯度被压制。

下一行动为0更新的同网格图像替换探针：确认已学答案依赖图像条件的程度，分别报告训练和富集错误的验证诊断，不把诊断替代原整体门槛。

语义补充：q20269服装logo支持Miller Lite，候选Bud Light有明确识别退化；q23461原10参考均74，但小号字体及catcher人物指代有不确定性，保留原74→24评分下降、收紧为语义待核。显示用原像素裁剪不等于实际模型输入，不作专家重标。

已登记[同网格图像条件探针](VLM_IMAGE_CONDITION_PROBE_V1_PROTOCOL.md)，0更新，不扩旧失败配置。
