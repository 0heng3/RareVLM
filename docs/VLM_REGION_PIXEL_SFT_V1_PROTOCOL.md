# 实际输入像素坐标SFT：匹配对照 v1

2026-10-20北京时间登记。旧结束符候选完整失败后停止，不接续其Final或取消EOS目标。

## 实测依据

0更新的固定64训练题，原归一化casefold exact23/64、补充尺寸归一化25/64、实际像素坐标41/64；像素对带尺寸归一化16纠正0新错，配对95%CI[+14.0625,+35.9375]pp。达到事前>=8净纠错/正区间准入门槛。这是同一权重上的输入适配收益，不能当模型训练或heldout能力提升。
Qwen2.5-VL技术报告§2.1.2/§2.2.1使用实际输入维度上的绝对位置。原0–1000框几何合法但不贴近原生表征，当前证据支持坐标输入是瓶颈之一，不认定全部旧失败均由此造成。历史仓库入口已转Qwen3，不冒称未取得的2.5 notebook已精读。

## 唯一研究因素

两组都给出processor实际宽高，完整原图、不裁剪、不画框、不输入目标文字。
normalized辅助：0–1000框＋实际尺寸；pixel辅助：实际输入像素框＋同一尺寸。
坐标从原points的原像素xyxy按实际image_grid_thw×14确定的宽高分别缩放，左上floor、右下ceil；保留官方文字、大小写和原1600图。两组只改变坐标单位及相应数值/单位提示，不同时改变EOS/LoRA容量或训练预算。

## 数据、训练和复用

原1600训练图/1600共同TextVQA QA不变；两组各新增1600辅助，总各3200唯一QA。按原区域SFT相同微批和图像顺序训练，1600共同＋3200辅助=4800唯一登记输入；1600共同CPU记录源锁复用，3200新辅助全部实际CPU编码，目标token与原官方标签逐值一致、输入<=2048/视觉<=1024，实际尺寸逐题核对。
直接复用冻结vlm_region_ocr_sft_v1.train.train，无训练循环复制或框架替换。两组均原SFT42起步、FP32 q/v LoRA1843200、BF16底座/FP32冻结head和CE、dropout0、TF32off、ordinary完整assistant CE（含结束符/尾部）、200更新/micro4/effective16、LR2e-5/20warmup、clip1/AdamW wd0。
原初始Text512/Human512预测按完全相同模型/数值/生成策略锁定复用；新normalized是本轮完整训练的匹配控制，不能拿旧不含尺寸提示的region替代。两组对main评测不提供GT框。等图/QA/assistant目标不等于输入token或实际计算完全相同，实际成本分别记录。
每臂训练active2700秒cap；首正式步保存LoRA逐值检查/完整实际输入trace，保留checkpoint恢复。GPU共享UUID锁、零compute/40GiB/三次15秒准入，不终止已有任务；一张卡可接续，不因另一卡占用而停工。

## 评测与门槛

Final@200固定，不在中间checkpoint看screen或选最好点。两组各Text512＋Human512，raw greedy/repetition1/max32/batch4/1024预算。随后两Final在同一原固定64像素提示上做训练内学习检查（新初始该口径41/64），分别报告raw/casefold exact，不把坐标变化前后混成权重增益。
pixel相对原initial和新normalized均须：Text整体及rare1–5>=+.5pp，整体6000次图像簇95%CI下界>0；frequent>20及Human overall/yes/no点差>=−.5pp。频次用原32985监督目标，Human按既有互补组件聚类；保留全部升降。旧人工+1pp/pair+2pp要求不改。
本轮单seed42/已曝光内部screen，属于探索；失败停止、不延长/扫参/挑checkpoint/写CV。若筛选全过，仍需独立确认及复核；本轮不自动追加训练、正式benchmark或种子。

## 边做边学

学习页给出原图、原始框到实际像素的计算、同模型128新生成与新训练结果。展示红框只用于教学，不是模型attention或实际图像修改。
依据：https://arxiv.org/html/2502.13923v1；代码src/vlm_region_coordinates_v1/data.py；零更新结果outputs/vlm_region_coordinates_v1/COMPLETION.json。4坐标几何检查已通过，完整新CPU编码另记录，不把实现验证当泛化证明。
