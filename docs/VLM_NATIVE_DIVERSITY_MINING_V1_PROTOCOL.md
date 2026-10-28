# 原生问答纠错数据覆盖：筛选 V1

2026-10-26，选择/CPU编码及新GPU预测前登记；与旧研究分目录保留。

已有结果：visual_preference_v1已完整5154生成/64更新，1617起点87.198516%、匹配CE87.303649%、CE+DPO87.631416%。候选对起点+.432900pp/95%[−.056866,+.957398]，对CE+.327767pp/[−.104362,+.798247]，未过原+.5pp/正区间门槛，不扩大旧配置。候选训练满分43/128，85残错中65仍为原negative、20换了回答；不能把负例过时当主要已证实原因。1617 rare342点差−.497076pp、multi469−.191898pp均为事后描述，无切片区间，不能用初筛rare366的+.546448混称长尾收益。

下一假设是相同QA曝光预算下的监督覆盖与重复次数取舍；当前先取得新可信纠错监督，**不宣称已证明数据规模为唯一瓶颈，不先增加DPO权重、epoch或换RL名字**。未来拟比较96独立纠错各2次与192独立纠错各1次（每组192纠错曝光），相同回放/总QA/32步及匹配优化；完整训练协议须在候选原图复核、数据准入后另行冻结，当前只允许筛选。

固定3072条：multiword224，singleword2848；数量依据无GPU输出的训练池库存6135 single/364 multi、去重且target cap2后2983/254。先按SHA(['native_diversity_mining_v1',qid])选224 multi，再选2848 single，图像/像素全局唯一、每normalized target≤2。只从原manifest train_all选，不读取新验证预测挑样。target至少8/10标注一致、原字符≤64/原词≤6，排空/yes/no/unknown/unanswerable/unreadable；排当前384训练target的380个归一化值。所有旧边界和两矿工3584图、当前训练/评测图均按image/pixel排除；不放松规则凑数。

样本为原始整图+原问题，无OCR文字注入，无GT裁图。使用native_scope共同复合初始adapter和同已验imagewise vision/FP32 language、FP32 adapter/head、1024视觉上限、greedy32/batch4。CPU仅新增3072条实际prefix/grid/token检查和原SHA记录，不重复旧加载/数值/梯度/251图审查。没有模型结构变化或零B移除，不新增8题结构parity。

双分片各1536QA，固定连续4题batch交错分片，0优化/0新reference/3072新自由生成。使用原空闲UUID共享队列、3×15秒确认、≥40GiB/<10%/零compute，不抢占其他任务。每分片≤1800秒、peak allocated≤65GiB；逐完整batch保存预测与actual prefix SHA，可续读；前后LoRA hash不变。CPU输入可从已保存前缀顺序恢复。

候选按SHA(['native_diversity_mining_v1:errors',qid])排序，只取自然EOS结束、原VQA≤.3、原normalized target不为prediction子串的条目，每target≤2。VQA只是筛选代理，**不是偏好真实性证明**。前256候选上限须原图复核；单列语义等价、格式/空格/日期/单位/翻译、问题范围和可见性不足，不改原GT/分数。

准入需求：新增≥64条图像支持且语义明确的错误，≥64个新normalized target；与原135支持条目合并后，预期至少可构造128 single+64 multi。数量不足则报告不足，不降低门槛或自动再挖池。本轮COMPLETION必须training_admitted=false，等待独立审核工件与新训练协议；不会因CPU通过、矿工退出0或候选数够就自动训练。未来主能力验收仍保留相对共同起点及匹配强控制各≥+.5pp、图像簇正95%区间和保持保护，单seed内部结果不当外部或稳定能力。

所有候选来自原SFT已曝光的train_all；仅在纠错数据池层面新增，预训练重叠未知。保留旧负结果、预测/源码/门槛；不改CV/科研/Agent/CI、不push。
