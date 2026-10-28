# 同图覆盖SFT完整结果与下一短正控

两组各200步/3200曝光已完整完成，控制训练2184.797秒、候选2159.305秒；实际曝光核对全部通过（unique2400/3200），无新增运行或曝光故障。

|模型|Text512|Human512|训练pair全对|
|---|---:|---:|---:|
|初始|85.859375|95.117188|19/64|
|repeat|85.898438|95.117188|26/64|
|dual|85.917969|95.117188|26/64|

dual对repeat只有+.019531pp，95%图像簇CI[−.333333,+.321932]跨0；rare差0。对初始overall+.058594pp、rare+.608696pp但CI跨0，frequent−.573770pp未守门槛。主联合失败，不继续这一候选。不能将训练loss差当生成优势。

同图覆盖增加800独立区域标签，在当前200步预算中未显示相对repeat的有效收益，不据此否定所有多区域训练。已核mask/梯度/保存/实际曝光的已有证据未发现新故障，不反复重审通过项目。

## 下一步为何是短学习正控

新标准为先测可信困难区域是否能学会，避免再直接启动200步。由初始模型完整图读错另一官方词、oracle裁剪能读对的训练例，经过几何/互非子串/不重叠过滤得16组候选；逐图审查后1组低对比/重音不确定排整组，1组B框含另一词中心排整组，准入14组双向偏好/28QA。拒绝词是实际同图混淆，A/B反向chosen/rejected，防止全局词偏好或长度捷径。不是人类RLHF标注或专家全池清洗。

CE已通过softmax对其他词产生梯度；新DPO只是把具体混淆序列放入显式相对目标，是否更好必须与同数据CE比较。复用vlm_alignment_signal.loss和原checkpoint工具，无新框架安装。reference为完整SFT42，不是禁用adapter后的底座。

32步正控：两组同数据/顺序/forward/LoRA和LR，8偏好chosen+8原VQA/步；偏好微批均chosen4+rejected4合批，SFT lambda0、SFT+DPO lambda1/beta.1，chosen完整CE保持。256独立回放＋28偏好QA，32步各512chosen曝光，参考按实际计划微批缓存并匹配padding hash。

初始及Final只做28困难QA+64本轮图像分离的原SFT已见保持题，均训练白名单，不叫heldout或主VQA提升。短正控门槛只决定后续研究，不放宽原main overall/rare+.5pp/正区间/保持门槛，不写CV；不按loss或相对reward宣布学会。

已提交新controller634880，先baseline/reference再sft/dpo_sft32步，不扩旧Final。模型能力结果以新COMPLETION.json为准。
