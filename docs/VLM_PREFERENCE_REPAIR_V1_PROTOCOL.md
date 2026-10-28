# 偏好语义审查及离线重挖协议 v1

2026-10-15。只处理 RareVLM 的50000份已生成训练回答，不修改原采样、评分、749候选或模型。数据集弱标签不等于直接人类偏好；助手看图审查不等于独立人审或用户掌握。

首轮审查在看图前保存 `vlm_preference_review_v1/AUDIT_PLAN.json`：749对中最低 SHA256(`rarevlm-semantic-audit-20261015:`+qid)固定64对；明确有效比例>=90%，Wilson双侧95%下界>=80%。不确定、问题歧义、同义及答案范围差异不计明确有效。未通过不训练原749对、不降低门槛。Wilson仅描述助手判断，不能解释成无误真值的总体偏好准确率。

## 机制修复：独立协议，旧失败保留

原规则先取第一对再过滤，会漏掉同组内更合适的负例；单向包含检查会漏掉被选答案只是较长正确答案子串的情形。复用固定10000图×4随机采样，检查所有可用正负组合，每图最多1对；候选chosen按十参考规范化票数、分数、原采样序号排序；rejected按原采样序号排序，选首个通过新规则的组合。不得生成随机假答案或把人工target冒充采样。

固定规则：沿用原质量检查；chosen支持从6提高到8/10；问题限定字面读数、名称、品牌、标题、作者等`literal_question`正则；排除rejected与任一参考的双向compact包含（两方至少2字符；数字也不例外，可能错删真实数字错误）；排除明确已知`coca cola/coke`别名。该别名字典来自首轮审查，仅覆盖一个已发现机制，不称通用同义检测；问题正则会漏收有效样本，也可能误收，8票也不能保证真值。源码为独立 `vlm_preference_review/repair.py`，旧 `quality.py` 与VQA官方计分保持。

新确认样本：排除首轮64个qid后，按最低SHA256(`rarevlm-semantic-confirm-repair-20261015:`+qid)选64对；同样>=90%明确且Wilson95%下界>=80%。该64未用于写规则；结果失败不得再据它调规则、换抽样或降低门槛。首轮64作为设计集单列，从修复训练候选中排除。此确认是数据质量抽查，不是独立模型泛化评测。新规则保留数/供给改变只是数据过程指标，不是提准。

每对保留源样本索引、来源锁、chosen/rejected与分数、票数。全量验证source_split=train、与dev/final图ID/文件及像素哈希不重叠；对本轮10000图保存生成与来源锁重新绑定。新manifest `training_allowed=false` 直到新语义确认结束且独立训练协议冻结。本轮不自动SFT/DPO、无optimizer更新，原结果不覆盖。

## 后续训练条件（尚未执行）

确认通过才冻结“继续SFT vs sigmoid DPO”：同一初始SFT seed42、同chosen/QA、同顺序与更新预算，DPO额外rejected/静态reference的token及算力单列。参考缓存必须来自固定初始SFT；不能每步对current actor.detach。assistant-only sum-logp与原结束token边界保持，SFT用每QA平均assistant CE；仅语言q/v LoRA1843200参数，基座/视觉冻结，新的BF16训练独立于历史NF4/FP32训练。

完整内部dev与rare/高频保护、Final固定预算、三seed顺序复核和旧资产保护需在训练前另行冻结；不按正式TextVQA/DocVQA/MMStar成绩挑超参数或checkpoint。不保证DPO一定优于继续SFT。[TRL DPO官方文档](https://huggingface.co/docs/trl/main/en/dpo_trainer)用于核对公式与reference约束，不安装新框架、不宣称完整框架复现。

所有新文件在持久目录；现有GPU任务不动。无CI、无自动push，不修改正式简历、论文或Agent。
