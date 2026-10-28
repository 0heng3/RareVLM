# 最新行动 2026-10-21 23:43

- 2026-10-21 23:43最新：`vlm_merger_transfer_v1`已22:05:28完整退出0/0争用；训练满分candidate64/qv56/初始0，seen512初始93.222656/候选93.085938/qv93.085938，原SFT未用图像dev1617初始87.328386/候选86.914038/qv86.951144，候选−.414348pp/CI[−1.205734,+.332318]，联合fail不扩该配置。多词469题8升(+4)/23降(−13.9)，净−2.110874pp/条件CI[−3.677419,−.589473]；13个降分缩短，但q1259增加AVN有原图支持不能称确定语义退化，q13944丢seasons有截短依据，评分保持不改。扩大merger在该预算没有迁移优势，不外推所有视觉适配无效。

- 新`vlm_reference_replay_v1`已执行v2控制器682161于23:32:53提交，23:33:26 plain_transfer682174→GPU0固定普通CE迁移评测/soft_kl682177→GPU1缓存校准训练，hard_replay待GPU0自然释放接续；保留其他任务。固定原qv1843200/原SFT起点5d8/expanded240同640QA同64步/512hard+512replay/高lr2e−4/FP32head与masters/dropout0。soft候选=L_base+βmean_replayKL(p_initial||p_actor)，hard强控制=L_base+同βmean_replayCE_GT；普通CE旧Final25425仅补dev1617、原三cohort复用，initial所有已有同帧预测复用；总6939新生成、128新正式更新，teacher512QA/校准48QA前后向另计。两域seen512/transfer1617分别对初始/plain/hard都需+.5pp/正95%簇下界及原rare/保持/训练净32等门槛，不以弱对照/子集挽救、不自动扩seed/官方final/CV。

- criterion闭式CPU oracle1项通过（KL方向/逐QA平均/左mask/梯度/同分布0），已有正确V2评测函数原样复用含optional frequency和末批1。准备冻结后核对发现任务plain_transfer与模型plain混用，0GPU时独立execution_v2显式映射、analysis_v2只重绑定模型角色列表，原函数体/科学源/PLAN不改；实际必须用tools/run_reference_replay_v2.py和queue_v2/STATUS/controller_v2.log，禁止旧入口。source/data已双锁、旧830训练CPU缓存及全部迁移输入元数据复用0新编码，原SFT图像隔离证明沿用。

- 已完成teacher128微批/512QA/1951assistant位置/151936全词表FP32，1,185,970,688 bytes，295.790秒，权重哈希保持；均值entropy .134694/非GT质量.034625（按QA再均值，非答案准确率）。只读取历史expanded控制step16/32/48及next block训练数据，0新optimizer：基础CE/KL范数比196.381194/10.149143/28.977664；按预定clip中位数β=28.977663982973098，未用验证分。KL对硬回放CE cos .626209/−.095986/.783308，三点相对差.945/1.081/.874，不是简单冗余；不据此证明β最优或Adam步长相等。校准34.699秒后精确恢复原SFT5d8并fresh Adam/RNG，非负checkpoint接训。首16实际输入/labels/baseCE差0、soft初始8KL全0已通过并开始正式更新；hard初始附加CE允许非0，不误套KL零限制。

- 预计2026-10-22 00:25–00:45全部训练/6939生成/分析（工程估计依赖资源，以实时文件为准）。cache600秒/校准180秒/每臂训练累计1800秒/评测累计2400秒/65GiB，stage PHASE区分缓存/校准/训练/评测，完整缓存/β/optimizer/RNG/checkpoint与UUID可恢复；不重复旧全局mask/权重/数值审查。协议docs/VLM_REFERENCE_REPLAY_V1_PROTOCOL.md及EXECUTION_V2，学习learning/教师KL保留与加强CE回放.html含8原图/损失/实际梯度校准/语义边界/后续模型真实回答。仅运行与信号检查通过，尚无新能力提升；用户目标未完成，CV/PDF/科研/Agent/CI不改不push。

# 2026-10-21 13:24：128不同可信错误已准入，匹配SFT已提交

3584实际生成已完成（0更新）：训练3072初始93.515625%、新512初始93.222656%，205自动候选。固定前184原图助手复核131支持/53排除或不确定，首128不同图/125不同target通过原至少两类各16/每target最多2准入；不把排除数当全池错标率，不改GT/评分。新671547两卡普通CE各64步：覆盖128对分层重复32、相同512困难曝光+512共享回放，16分层逐slot匹配；1调度合同与640CPU过，每步分源指标追加、optimizer/RNG可恢复。初始预测同batch4全部复用，Final各1044实际生成/只128训练题评分。候选对初始/控制各+.5pp与正CI、rare/保持/训练净32联合门槛，不自动扩长或主测/CV。[新协议](VLM_COVERAGE_SFT_V1_PROTOCOL.md) · [本轮结果](VLM_COVERAGE_SFT_V1_RESULTS.md) · [128原图与重复子集](LOCAL_ARTIFACTS.md)。下方为历史。

## 2026-10-21 11:54：纠错曲线完成，转向扩大不同错误监督

原单轮GRPO联合失败（93.828125%对新CE94.023438%）；事后语义复核不重评分。已存检查点曲线256新生成/0更新完成：SFT满分1/2/5/5/6/5，GRPO1/2/4/7/7/6；26/25题在观察点未曾满分，各1回退，停止在同32题换损失或回选checkpoint救结果。

669962已提交新3584原生QA挖掘：3072新训练候选＋512预固定验证，3584唯一pixel/CPU编码完成/2批次合同过；原始SFT、0优化器，完整四题批边界分片并保留真实token/EOS。监督频次明确扩大1–20并分层，不混旧rare口径。至少128原图可见支持的不同错误及类别/target多样性才另立强SFT覆盖对照，不自动长训/RL/主测/CV。[纠错曲线](VLM_GRPO_LEARNING_CURVE_V1_RESULTS.md) · [原图与评分边界](LOCAL_ARTIFACTS.md) · [新挖掘协议](VLM_ERROR_COVERAGE_MINING_V1_PROTOCOL.md) · [新结果](VLM_ERROR_COVERAGE_MINING_V1_RESULTS.md)。下方为历史。

## 2026-10-21 10:29：单轮GRPO × 新强SFT对照已提交

10:38更新：数值准入已过（535动作ratio100%在clip内、PG梯度2.113456），新512初始93.691406%。两卡正式CE7/32、CE+GRPO4/32，首16输入一致/初始CE差0；约10:50–10:56完成训练和配对评测。尚无Final能力结论，按实际PROGRESS为准，不重复已通过审查或自动扩训。

数值隔离与256采样均完成：17/32清晰正负组、43满分、0更新。原图复核17组后，666332控制器已提交新研究：32步普通CE对32步CE+GRPO，同512 chosen/顺序/起点，候选额外136实际动作只用一次，clip .2/λ1/KL0。先17组行为概率路径准入与新512初始评测，失败停止不带数值偏差训练；通过再两卡训练。新512独立于旧矿池/各诊断同像素，旧32+64初始同条件复用，单seed训练白名单内部筛选，不称盲测/多轮在线RL。候选须对初始/新CE均+.5pp及正区间，并过切片/保持/训练净8门槛；0自动扩长/主测/CV。[预登记](VLM_GRPO_ITERATION_V1_PROTOCOL.md) · [本轮结果](VLM_GRPO_ITERATION_V1_RESULTS.md) · [17原图/动作/奖励优势学习页](LOCAL_ARTIFACTS.md)。下方为历史。

## 2026-10-20 23:57：真实动作接口已修正，下一步采样信号实验已提交

新`vlm_sampled_signal_v1.trajectory`不将生成文本重分词当实际动作、不伪造EOS/加入postEOS模板换行，裁头保留最早监督位置的前一个预测位。4项CPU验证及32真实前缀/96历史动作保持通过；6/96重编码差异来自3模型的同两题，不外推全部失败。控制器656785已提交原始SFT42对既有32训练题各8次/共256采样（T1/p1/k0、0更新），先等655469数值隔离完成再共享GPU准入。事前至少16题有自然EOS的满分/<=.3正负，才考虑在线优化；不足停止本轮设置，不加组数/温度/seed凑结果。通过亦需新采样语义原图复核、独立匹配SFT和在线学习协议，不自动训练、改简历或回写旧512失败。当前尚无新采样/模型提升。[固定协议](VLM_SAMPLED_REWARD_SIGNAL_V1_PROTOCOL.md) · [结果与实时输出入口](VLM_SAMPLED_REWARD_SIGNAL_V1_RESULTS.md) · [真实原图/BPE/奖励学习页](LOCAL_ARTIFACTS.md)。下方为历史。

## 2026-10-20代码级定位：真实决策差距已测量，数值隔离排队

96实际生成+192训练序列前向/0更新完成；原生成全复现、初始reference残差0。DPO23题仍未满分，23题relative均改善但22仍绝对序列劣于负例。换行占分母21%但CE0，仅为缩放限制；canonical字符串重编码在2/29负例中不等实际token轨迹。teacher与greedy有临界符号翻转；655469单例8条件只读隔离正等其他GPU任务结束，不抢占，未获得dtype因果结论。[完整分因](VLM_ERROR_DECISION_TRACE_V1_RESULTS.md) · [32图/token/概率](LOCAL_ARTIFACTS.md) · [隔离状态](VLM_ERROR_NUMERIC_PATH_V1_RESULTS.md)。旧训练/512失败门槛不回写，无新训练。

## 2026-10-20 21:24：双问诊断完成，未观察新增跨图迁移

128新图/256QA×3旧模型已完成，768生成/0更新。三模型VQA91.484375%、两问满分106/128、同答2/128，原始输出逐字全同；不能由[0,0]区间推总体等价。普遍忽略问题假设无足够支持，不直接加双问训练；公交是局部选错车辆，球衣存在口径歧义。原CE/DPO512门槛失败不回写、不扩训/主测/CV。后续新训练优先有效原生错误覆盖或可验证奖励；若考虑GRPO须先验证同题采样有可靠正负及组内差异，目前未采样/未训练。[完整新诊断](VLM_NATIVE_QUESTION_BINDING_V1_RESULTS.md) · [128原图与公式](LOCAL_ARTIFACTS.md) · [CE/DPO结果](VLM_ERROR_SUPERVISION_SFT_V1_RESULTS.md)。下方为历史。

## 2026-10-20 21:15：真实错误CE/DPO完成未过门槛，原生双问零更新诊断已提交

SFT验证+.449219pp/DPO+.253906pp、区间跨0，原联合门槛均失败；DPO多记住2训练题未变为迁移优势。停止该分支扩训，不降门槛。原图可见Close→N97目标范围错误，其他收益多为位置/名称补全，不能称普遍OCR提升。新128图原生双问/256QA×3已存模型、768生成/0更新，649140在共享队列运行；QA及两问均对一起判断，输出变化率不等于正确性，不自动接训或改简历。[完整CE/DPO结果](VLM_ERROR_SUPERVISION_SFT_V1_RESULTS.md) · [双问固定协议](VLM_NATIVE_QUESTION_BINDING_V1_PROTOCOL.md)。下方均历史。

## 2026-10-20 19:01：CE/DPO实际训练健康，等待Final独立验证

reference64/64完成，CE4/32和DPO3/32正在两张A100运行；首步输入/CE完全相同、相对reference margin0、梯度与FP32保存通过，复用STARTUP_HEALTH不重复首步审计。真实起点训练满分1/32、验证93.183594%、保持93.75%；未出候选Final，不能声称提分。预计19:17–19:23含训练和608题/臂生成完整结束，以实际状态为准。[实时与Final结果](VLM_ERROR_SUPERVISION_SFT_V1_RESULTS.md) · [数据准入与失败原因](VLM_ERROR_SUPERVISION_MINING_V1_RESULTS.md) · [原图和实际reference公式解释](LOCAL_ARTIFACTS.md)。下方为历史。

## 2026-10-20 18:50：真实错误32对已准入，CE/DPO匹配对照已提交

1536生成结束；1024训练池76候选，52逐图审查得到34支持/18排除，按顺序取32。排除了Visual Studio有效简称、9,95忠实转写等低分伪负例；不改原GT/评分。320真实监督编码通过、训练/验证/保持无像素重叠；645835先reference再两臂各32步，512独立验证初始93.183594%，候选完整结果待出。旧EOS5<32不改变、不把新错词算EOS。[完整数据结果](VLM_ERROR_SUPERVISION_MINING_V1_RESULTS.md) · [原图与CE/DPO原理](LOCAL_ARTIFACTS.md) · [训练固定协议](VLM_ERROR_SUPERVISION_SFT_V1_PROTOCOL.md)。下方为历史。

## 2026-10-20 18:27：原始SFT真实误答挖掘已启动

独立1536图/QA（1024训练池+512预固定验证，各半单词和多词），1536 CPU真实编码完成；644588控制器、两片各768实际greedy，无teacher/EOS代理、无退化模型接训。原8票/质量差.7/短答<=.3保持，最多审查前64、需32可见证据支持；不足不训练，不把新错词凑入旧EOS门槛。同数据CE与CE+DPO32步协议已事前固定，尚未准入训练，不将挖掘量当提分。[新挖掘协议](VLM_ERROR_SUPERVISION_MINING_V1_PROTOCOL.md) · [条件训练对照协议](VLM_ERROR_SUPERVISION_SFT_V1_PROTOCOL.md)。下方为历史。

## 2026-10-20完成：真实错误挖掘发现5对可信截短，停止自动训练路径

512候选/676词边界已筛选，固定128真实greedy已结束；112满分，6真实前缀截短中5对达到质量差且已助手逐图复核支持。5<32，数量门槛失败，0优化器更新/新adapter，不启动SFT/DPO、不降阈值。其余10条低分非前缀错误尚未全部逐图复核，单列保存，不凑EOS数量。长答案比例实验无稳健提升，下一只考虑另登记按真实错误类型收集有效监督，再独立匹配完整CE与CE+DPO；不直接续训负Final、扩步数或换RL名称。没有新增简历提分。[完整结果](VLM_SCOPE_PREFERENCE_MINING_V1_RESULTS.md) · [五张原图与EOS原理](LOCAL_ARTIFACTS.md)。下方均为历史快照。

## 2026-10-20 17:19当前：真实EOS截短偏好训练池挖掘，0更新

512诊断完整：初始94.316/3截短，control93.848/5，balanced94.043/5；比例候选无稳健优势，停止加训/扫比例。78新增题原已73满分/0截短。已启动643197：独立train稀有多词512图，676边界teacher筛选后实际验证top128，仅score<=.3且GT高>=.7的真前缀作候选；数量>=32也不自动训练，需可见证据与独立匹配SFT/DPO。本轮不回填诊断题，不新增框架/adapter。[完整结果](VLM_SCOPE_TRANSFER_DIAGNOSIS_V1_RESULTS.md) · [EOS图片与原理](LOCAL_ARTIFACTS.md) · [固定协议](VLM_SCOPE_PREFERENCE_MINING_V1_PROTOCOL.md)。下方为历史。

新增实际证据：initial在78条新增长训练QA上73题VQA满分、前缀截短0。未来训练优先对准另行划分训练池中的真实错误/有可靠质量差的监督，不能只按答案长度采容易题；本次512诊断集不得回填训练。当前只有已存模型零更新诊断，无新训练。

## 2026-10-20 16:52当前：长答案有效信号和512分层迁移的0训练诊断

匹配SFT v2完整结束，两组区域11/28、保持60/64、多词31/32、截短0，未过门槛。实际多词67/145生效；新回放CE低/64题无目标失效事件，不能据零差判总体等价。已提交controller642559：3个已存模型各512分层多词QA，initial另78新增训练长问题，590CPU前缀过/1614预测/0训练。区分新增监督已会和迁移不足，不改旧门槛或自动加训/主测/CV。[完整结果](VLM_SCOPE_BALANCED_SFT_V2_RESULTS.md) · [诊断固定协议](VLM_SCOPE_TRANSFER_DIAGNOSIS_V1_PROTOCOL.md) · [实际图解](LOCAL_ARTIFACTS.md)。下方为历史。

## 2026-10-20 16:30当前：长答案回放匹配SFT修正版在跑

v1顺序映射形成块状课程，19/15步主动停止、保留原记录，不作为完整算法结果。v2独立固定shuffle共用图像顺序，多词分散至四阶段；同67/145长答案比例、同SFT42/普通CE/32步。controller641914，两卡实际训练，首正式输入/共同区域CE/保存过。初始92题已完成同字节复用，新保持0训练图重叠；约20秒/步，估16:41–16:44完成，最终收益待实际生成。不自动扩训/主评测/CV。[修正版协议](VLM_SCOPE_BALANCED_SFT_V2_PROTOCOL.md) · [真实同图问答＋token＋代码](LOCAL_ARTIFACTS.md)。下方为历史。

## 2026-10-20 16:15当前：同图长答案回放匹配SFT已提交

controller641211：两组原SFT42新训32步，普通完整CE，同256图，67/145多词回放（全曝光13.09/28.32%），只在78槽换真实问题与完整GT。旧保持13图重叠已避开，新32short+32multiword保持64真实CPU前缀过，0本轮训练图重叠。复用可信训练/保存/恢复与334回放编码，不新增R/额外token权重/teacher。Final只28区域+64新保持；按预定短门槛看纠错/截短/保持，不凭loss或数据通过当能力修好，不自动主评测/长训/CV。[固定协议](VLM_SCOPE_BALANCED_SFT_V1_PROTOCOL.md) · [上一轮主任务失败](VLM_PLAIN_SFT_MAIN_V1_RESULTS.md)。下方为历史。

## 2026-10-20 15:04闭环结果与下一步数据修正

所有当前GPU验证已完成：普通SFT有区域迁移122→133/256，但原始Text85.859→85.488/Human95.117→94.922，主门槛未过；额外残差没有强基线收益，停止模块和token加权。4有损前缀截短损失2.3score，训练多词曝光13.09%显著低于原去已知固定响应后28.42%，支持粒度迁移取舍假设，未证唯一病因。已实际准备同256图的67/145多词匹配回放视图，334CPU编码通过，状态明确数据准备/未训练；新保持集需排训练重叠，下一独立登记普通CE匹配对照，不扫参/盲长训/CV。[完整主任务分析](VLM_PLAIN_SFT_MAIN_V1_RESULTS.md) · [数据修正设计](VLM_SCOPE_BALANCED_REPLAY_V1_DESIGN.md) · [全部原图与解释](LOCAL_ARTIFACTS.md)。下方为历史。

## 2026-10-20 14:58当前：普通SFT的主VQA零训练筛选

四条件128图迁移完成：初始122、普通SFT133、完整读出133、残差关闭134/256。读出模块无额外价值，停止模块分支；普通SFT11修0破、图像簇区间正，6修是范围控制、5换错词，尚不是主VQA能力增益。已提交controller640218，普通SFT32只新增Text512/Human512预测，0训练、复用同策略初始；按原+.5pp与保持标准检验，不回调参数或简历。[完整结果](VLM_READOUT_TRANSFER_V1_RESULTS.md) · [全部128图互动解释](LOCAL_ARTIFACTS.md) · [主任务协议](VLM_PLAIN_SFT_MAIN_V1_PROTOCOL.md)。下方为历史。

## 2026-10-20 14:44当前：读出模型0更新迁移＋直接残差消融

位置实验读出12/28/pair2、视觉10/28/pair0、普通11/28/pair1，保持57/64。视觉候选停止；读出是控制中的事后微弱信号，先不扩训。已提交controller639428：128图/256QA，与本轮训练/回放/保持及指定旧诊断图像分离，四条件1024预测＋28完整重载核对，0更新。不是初始SFT未见或正式benchmark。依据新协议分别判断迁移和额外模块即时贡献，不据旧训练题或保存成功写CV。[完整位置结果](VLM_REGION_ADAPTER_LOCATION_V1_RESULTS.md) · [固定迁移协议](VLM_READOUT_TRANSFER_V1_PROTOCOL.md)。下方为历史。

## 2026-10-20 13:59当前：等参数残差适配位置对照

关键token候选9/28落后整题加权与普通SFT11/28，停止损失加权分支；完整实际输入和loss公式无新增故障。两组新32步已提交（controller638040），相同131072新增参数分别放视觉输出/语言读出，总1974272，普通完整CE与原数据/初始化/顺序不变。CPU恒等/梯度过，GPU与Final收益待验证；新模型需同时保存LoRA与残差，不自动长训/主评测/简历。[上一轮结果](VLM_REGION_DECISION_SFT_V1_RESULTS.md) · [实际图片与源码图解](LOCAL_ARTIFACTS.md) · [新固定协议](VLM_REGION_ADAPTER_LOCATION_V1_PROTOCOL.md)。下方为历史快照。

## 2026-10-20 13:39当前行动：关键token监督匹配对照

只读6QA/36序列/0更新已完成，TELL完整CE下降而首决策CE上升，支持检验监督分配取舍，不认定唯一病因。两组新32步已提交（controller637495）：20固定训练错题上额外完整CE对额外首分歧token CE；同初始/输入/顺序/预算，40新CPU编码已过。普通SFT32复用，新等QA权重控制实际训练。只看28+64训练题，过预定门槛才另议扩大，不改简历或主VQA标准。[完整只读结果](VLM_REGION_DECISION_DIAGNOSIS_V1_RESULTS.md) · [实际图片与源码](LOCAL_ARTIFACTS.md) · [新协议](VLM_REGION_DECISION_SFT_V1_PROTOCOL.md)。下方为历史记录。

## 2026-10-20 11:39当前结论：短SFT/DPO正控完成

区域QA8→11/28、pair0→1/14，CE与CE+DPO相同3纠正/0新错，保持57/64；未达到预设短学习准入，DPO无额外收益。已完成训练和离线margin诊断，不扩长或扫参，不写简历。下一优先对可信未纠正样本检查首个分歧token和q/v/视觉接口梯度覆盖，再确定最小受控实验；未声称新GPU诊断已执行。见[VLM_REGION_PREFERENCE_PROBE_V1_RESULTS.md](VLM_REGION_PREFERENCE_PROBE_V1_RESULTS.md)与[真实图解](LOCAL_ARTIFACTS.md)。下方启动时间和PID均为历史记录，以本项和COMPLETION为准。

2026-10-20当前：区域偏好32步学习正控已提交，SFT与SFT+DPO同数据/forward，14双向pair，完整SFT42参考缓存；先验证可信难例学习，不自动长训或写CV。上一同图覆盖主任务失败。见[VLM_REGION_PREFERENCE_PROBE_V1_PROTOCOL.md](VLM_REGION_PREFERENCE_PROBE_V1_PROTOCOL.md)。

2026-10-20当前：同图区域覆盖SFT已提交，800图、repeat/dual各200步/3200曝光，实际unique2400/3200；旧负候选不扩训。坐标切换pair19→27/64，已有框依赖但精度不足。见[VLM_REGION_BINDING_SFT_V1_PROTOCOL.md](VLM_REGION_BINDING_SFT_V1_PROTOCOL.md)。

2026-10-20最新：256图/1024条迁移诊断完成，主要收益是范围收紧，100残错45同图另词。1389组同图双框已构造，固定64组坐标切换零更新诊断已提交。当前不扩旧长训；见[VLM_PIXEL_TRANSFER_DIAGNOSIS_V1_RESULTS.md](VLM_PIXEL_TRANSFER_DIAGNOSIS_V1_RESULTS.md)。

2026-10-20最新：像素坐标训练完整未过门槛；64题的+4净正确主要是范围变化。新256图完整/GT裁剪零更新迁移诊断已启动（1024生成），待其分型后选择新算法；不扩旧200步。见[VLM_REGION_PIXEL_SFT_V1_RESULTS.md](VLM_REGION_PIXEL_SFT_V1_RESULTS.md)。

2026-10-20最新：结束符消融完整失败；零更新定位坐标约定适配，casefold25→41/64。新匹配像素坐标训练正在CPU准备，两臂完整CE，不混用GT框收益和无框VQA能力。见[VLM_REGION_OCR_TERMINATION_V1_RESULTS.md](VLM_REGION_OCR_TERMINATION_V1_RESULTS.md)及新协议。

2026-10-19最新：区域SFT完整负结果；固定训练任务有学习但多词回答出现截短。已启动仅移除区域辅助结束符监督的200步匹配消融，详见[VLM_REGION_OCR_FAILURE_AND_TERMINATION_V1.md](VLM_REGION_OCR_FAILURE_AND_TERMINATION_V1.md)及新协议。不得重启旧controller619691，不改简历；以新输出状态为准。

# JD驱动的下一步：图文召回后的重排

2026-10-19区域文字监督已准备并提交自动队列，controller619691；这是启动快照，当前以`outputs/vlm_region_ocr_sft_v1`的baseline_queue/training_queue和训练PROGRESS为准，勿重复启动。[固定协议](VLM_REGION_OCR_SFT_V1_PROTOCOL.md) · [真实原图与区域监督学习页](LOCAL_ARTIFACTS.md)。1600匹配训练图、每臂3200唯一QA/200更新，7新CPU项和4800真实输入编码通过；新基线/64题训练信号准入后才训练，已有GPU任务保留。当前尚无本项能力结果。

2026-10-19论文/源码检索建议：[增量训练瓶颈与优化顺序](VLM_FRONTIER_BOTTLENECK_REVIEW_20261019.md)。优先验证训练白名单上的区域文字真值监督，先与普通继续SFT匹配比较，再决定图像条件DPO、有效奖励GRPO或单独视觉适配；不是已准备/已启动研究，不降低旧门槛，不把论文或局部NLL增益写成项目提分。6篇核心论文与10仓库26文件归档可复核，无新GPU训练或环境改动。

2026-10-19最新SFT故障定位入口：[真实Adam更新、数值效应与局部修复](VLM_ACTUAL_ADAM_AND_NUMERICS_V1.md)。首步复算梯度与保存动量相对误差6.66e−8；完整FP32参考解释BF16小步一阶预测失真，但可信no到200步仍退化。独立FP32输出头与Adam位移约束候选已实现、6CPU项通过，局部no NLL改善尚未转成4题纠错；0正式训练/新adapter，不据局部loss门槛恢复旧分支或长训。后续数据用587对/1174QA新视图；新数值策略必须匹配初始/控制/候选。本页下面是历史CLIP计划与完成边界，不作为当前待启动状态。

本页保留2026-10-14执行前的候选方案。用户随后授权执行，已完成固定权重校验、原生预检与完整dev/test重排：test文搜图+17.273pp、图搜文+15.233pp，0新训练；当前结果、成本和边界以[完成报告](CLIP_RETRIEVAL_RERANK_V1_RESULTS.md)为准。下文的建议规则与进入条件为原设计，不再表示当前待启动状态；保留CLIP/Qwen主线，不新增重复项目。

## 为什么优先重排

- [1688团队招聘](https://liamding.cc/hiring.html)的工业VLM方向明确包含品商Embedding与Rerank；杭州研究实习入口同时覆盖多个研究方向，不能把整页当作多个独立实习JD。
- [阿里国际站多模态实习原文](https://career.nankai.edu.cn/correcruit/content/id/118629.html)要求图文文档表征与检索、全链路优化、延迟和吞吐。图文检索和DocVQA分别已有实验，但文档问答不等于文档检索，离线QA/s不等于在线服务。
- [海康杭州多模态实习](https://talent.hikvision.com/home/socity/position?postId=E916B963BC9E74350370DF43D2D2A8A3)侧重图文数据与算法复现；后续数据质量选择可作为针对该类职责的分支。
- [字节Seed官方方向公告](https://seed.bytedance.com/zh/blog/bytedance-seed-2027-foundation-model-campus-recruitment-is-now-open-internships-included)包含多模态预训练、视觉推理、生成理解统一和机器学习系统，属于更宽研究方向，不视为已核验的杭州日常实习JD。
- [腾讯杭州多模态数据岗位](https://careers.tencent.com/jobdesc.html?postId=2083008230946160640)要求数据生产、跨模态检索和高吞吐服务，明确两年以上经验；用于技术路线参考，不当实习资格匹配证据。

此前定向统计为5条杭州相关实习线索（阿里4、海康1；含两份方向公告），检索宽类3/5，明确Rerank与文档检索各1/5；另4条腾讯社招单列。样本非随机、偏阿里、部分是方向公告，不代表整个市场热度/HC，也不证明当前可投。毕业年月、办公城市和开放状态须投递前按具体JD核验。

## 已有可复核起点

| 已完成能力 | 数字与边界 |
| --- | --- |
| 图文Embedding适配和教师蒸馏 | 固定Flickr30k测试协议，T2I R@1 61.640→67.820（+6.180pp），I2T74.533→79.833（+5.300pp），三seed；不是电商线上收益 |
| 蒸馏机制对照 | 原蒸馏减负项置乱dev T2I+.800pp、区间[.227,1.393]；I2T+.267、[−.733,1.300]跨0；联合门槛未过，没有新test |
| 离线推理优化 | DocVQA同checkpoint强配置确认，阶段吞吐中位2.385×，ANLS91.843→91.841；仅该任务通过，不是线上SLA或全任务无损 |

来源：[蒸馏指标](../results/clip_retrieval_distill_v1_metrics.json)、[机制结果](CLIP_RETRIEVAL_SHUFFLE_V1_RESULTS.md)、[推理指标](../results/vlm_strong_confirm_v1_metrics.json)。保留成功与负结果，不用新实验掩盖已完成能力，也不把源码执行视为个人掌握或独立贡献。

## 候选实验与进入条件

1. 固定现有三个distill Final@10 checkpoint，复用完整dev1000图/5000描述的冻结CLIP特征，先生成Top-50召回。报告R@50作为该候选池的命中上限；若正项未召回，重排无法救回。
2. 核验[Salesforce/blip-itm-base-coco模型卡](https://huggingface.co/Salesforce/blip-itm-base-coco)及对应实现、权重revision/SHA、processor与ITM正类索引。官方卡注明COCO匹配训练；仍需记录预训练和数据重叠未知，排除在本Flickr30k划分专门微调的权重。参考BLIP图文匹配机制；不强制整体迁移LAVIS。
3. 对照仅为同一Top-50候选池的原CLIP排序与直接ITM匹配排序，不扫混合分数权重。图像与文本联合编码帮助判断候选关系；新增模态计算、数据来源和计算成本必须单列，不能把收益全部归于“等计算的新算法”。
4. 两方向各完整dev评测三已有模型：每seed最多(1000+5000)×50=300000对，总计最多900000次逻辑候选匹配。可跨seed按源图ID/文本ID去重缓存，实际唯一pair数和实际执行数分开；三个旧checkpoint不是三次新重排训练。
5. 先64pair×3重复做吞吐/显存预检，再依据实测估计全量耗时；不预报准确率提升。按共享UUID锁与连续空闲采样使用A100，不终止或抢占已有任务。

## 建议的验收规则（实施前另立冻结协议）

T2I为主提升目标：相对同checkpoint原排序R@1至少+1pp、图像簇校正单侧下界>0；I2T以−.5pp为可容许损失，要求条件非劣。两端点alpha=.05/2，三seed原始值/样本SD与固定checkpoint平均后的条件图像CI分开。nDCG、候选池Recall上限、完整预处理/编码/匹配/排序耗时、显存allocated/reserved与失败率一并报告。dev门槛失败保留结果并停止，未通过不新评test；历史test曝光须注明，不称盲测。

新增质量和成本在实测后共同判断，不能只看R@1。具体资源上限、延迟容许值、模型输入和检查点恢复合同在权重/64pair预检后形成独立正式协议，不据完整dev/test回调阈值。

## 学习、简历和其余方向

学习按“真实查询图和Top-50 → 双塔与联合图文匹配的张量 → ITM分数如何改变排序 → 命中上限、失败例与质量/延迟结果 → JD业务追问”推进。重点掌握现有对比学习、多正目标、KL、三seed/图像簇统计与当前代码。

简历可在用户掌握并确认实际贡献后，针对检索/训练/评测JD分别突出既有成果；不等待重排完成才投递，不以助手工作自动补个人贡献。传统CV版本保留长尾证据，多模态版本按岗位选择已有SFT/检索结果；本轮不改正式简历。

数据质量选择、文档检索、在线推理服务按明确目标JD另排期；RL/DPO、音视频生成、世界模型和大规模CPT目前不并行扩展。若面试反馈指出对应缺口，再依据真实岗位需求调整。

## 最新执行：曝光受控hard＋蒸馏已完成

新增六次完整训练，固定三组dev对照。图搜文相对随机蒸馏+1.267pp，但文搜图+.127pp未达+1pp主门槛，联合失败、未新增test；[完成报告](CLIP_RETRIEVAL_CAPPED_V1_RESULTS.md)保留源锁、统计与边界。后续按已讨论顺序推进自适应重排序，先固定校准/确认与真实pair执行合同；该项未启动。数据质量SFT仍须由错例和目标JD决定，不自动增加训练轮数或更换底座。

## 最新执行：自适应重排已完成

固定cal500/confirm500，0新模型训练。实际pair−21.88%、cached matcher＋route中位1.277×，双向R1 .5pp非劣通过；30%成本目标未达，联合失败、不新增test。见[完成报告](CLIP_RETRIEVAL_ADAPTIVE_V1_RESULTS.md)。如果继续研究，先在cal诊断每query Top20/50候选预算保留率，再独立预注册分层预算；本轮确认集不用于回调τ。候选尚未执行，SFT数据质量仍由具体错误和JD需求决定，正式简历保持。

## 最新执行：候选预算已完成固定test

0／20／50预算实际test pair−32.59%、离线缓存matcher＋route1.476×，双向.5pp非劣通过；T2I下降.207pp，未声称无损／提准。旧“预算尚未执行”是历史候选，现在以[完成报告](CLIP_RETRIEVAL_BUDGET_V1_RESULTS.md)和[JD对齐](JD_EVIDENCE_MAP.md)为准。当前项目已有表示学习→两阶段检索→质量成本优化的连续证据；下一优先是独立讲清核心函数与真实失败例，并按具体JD选已有成果，工业中文数据／文档检索／数据质量训练须明确需求再立项。正式简历保持。


## clip_retrieval_quality_v1完成更新

clean dev T2I+0.227pp，全门控失败未新增test；noisy dev T2I+1.160pp，全门控失败未新增test。12次完整训练、匹配4对4监督与采样；known mismatch诊断不当原始语义错标。失败策略不自动扩增。下一步先用真实图片、select_slots、selected_terms和独立评分解释成功／失败与历史full5差距；不因该实验堆更多项目，不将掌握未确认内容写入正式CV。


2026-10-15后训练推进：[JD分层统计](POSTTRAIN_JD_DECISION_20261015.md)、[信号审计结果](VLM_ALIGNMENT_SIGNAL_V1_RESULTS.md)。完整10000训练图／50000回答已完成，1480原偏好对、749保守候选。固定助手看图审查50/64明确（78.125%，Wilson95%下界66.567%），未过90%与下界80%门槛。独立机制修复从四个实际采样重挖217对，排除设计集18对后199对；不同64图确认56/64明确（87.500%，下界77.225%），仍失败；不再据该确认集调规则／换抽样。两个不同总体的比例不能当配对改善。

本轮0新GPU生成／0optimizer更新／0dev-test预测，原749与修复199都未启动训练；保留同义、范围歧义、参考与图中文字冲突的证据。六合同及来源／原采样／128图学习页／简历保护核验通过，[完成报告](VLM_PREFERENCE_SEMANTIC_REVIEW_V1_RESULTS.md)与[指标](../results/vlm_preference_semantic_review_v1_metrics.json)已归档。下一候选先建立有逐例图像依据或公开RLAIF-V的独立偏好数据协议，审核AI反馈标签、TextVQA等来源重叠、长短答案及prompt，再冻结静态reference的继续SFT／DPO对照；尚未下载或训练公开数据。岗位方向保留，不为补DPO关键词训练不可靠偏好，不改正式简历。

## 2026-10-15公开偏好下一行动：噪声诊断已启动

选择更小且有人类局部纠错来源的RLHF-V而非直接上83k AI反馈：5733原对，排121，一图一对后4007，train3568/heldout439，排既有各split/正式集与POPE的文件/像素重叠。原64明确49/64，数据事实质量门槛仍失败，DATA_ADMISSION拒绝保留；不把“human”当无噪声真值。

基于这项证据另预注册[公开噪声偏好诊断](VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_PROTOCOL.md)，使用全量3568对比较同起点SFT/DPO，不训练原749/199或64审计集，不降低原门槛。单独diagnostic-only准入、禁止自动晋级/更多seed/正式集/简历。两GPU预检通过且精确恢复临时更新，64 CPU合同通过；控制器349070已在两空闲A100开始SFT训练和DPO静态reference缓存，各一epoch/223正式更新。当前性能pending，观察窗口为Text1617、POPE9000/500图与heldout439的概率诊断；老师强制概率排序不是生成准确率。[启动快照](VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_RESULTS.md)与新STATUS为准，原未下载/未训练说明保留为旧阶段历史。

## 公开偏好诊断完成与失败分析

上述running为历史。GPU工作流17:00完成，SFT/DPO各223更新、三模型4851 Text＋27000 POPE＋1317 heldout pairs全部完成；独立CLI补报只恢复汇总，0重训练/重预测。Text初始87.112→DPO86.840（−.272pp，条件95%CI[−.574,−.006]），POPE87.1保持，提准与Text .5pp保持均失败。

[失败分析](VLM_PUBLIC_POSTTRAIN_DIAGNOSTIC_V1_FAILURE_ANALYSIS.md)以权重/概率/真实图片和上游源码为依据：LoRA BF16舍入后86.09%元素改变，DPO heldout74.03%相对margin改善，训练非空跑；Text仅21原答变化/8分变，POPE少3FP同时多3FN。初始POPE错误89.41%为漏检，优先改短答/物体正例监督及当前模型错误覆盖，再单独检查差异token加权。上游发布weighted SUM gamma1.1与本地uniform SUM不同，无因果消融；若后续只新增加权一臂，需正确重存per-token reference，不同时改mean/beta/lr/数据。当前没有新实验启动，数据事实质量gate与自动晋级false保留，正式简历保持。
