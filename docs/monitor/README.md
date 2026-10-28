# 实验观察台

当前默认研究由`outputs/rarevlm_monitor/ACTIVE_STUDY.json`登记为`vlm_native_scope_v1`：两组零更新scope预检→共同新基线960→语言/视觉LoRA各32更新→内部1920/条件三模型1617。`--run-root`可显式查看旧研究。当前仅标准CE，曲线与角色按研究隔离；旧SFT/KL和完整梯度诊断保留历史记录，未测指标留空。

覆盖RareVLM当前原生问答纠错研究的数据筛选、审核、固定参考、训练、内部评测、条件迁移、GPU进程和错误证据。它只读取工件，不加载模型、不重评分、不恢复或终止训练，也不改动冻结实验源。其他项目的GPU进程仅展示为“其他任务／未关联”。

每次检查运行：

```bash
source /opt/data/private/.tools/activate.sh
source /opt/data/private/.tools/vlm-a100-env.sh
"$VLM_A100_PYTHON" /opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001/tools/rarevlm_monitor.py --snapshot --render
```

生成工件在`rarevlm_pilot/outputs/rarevlm_monitor/`，不进入科学实验目录：

- `状态看板.html`：交互快照，切换指标、损失，搜索案例，查看图像和原始记录。
- `总览.png`：一次检查的状态总览，回复中应直接展示，并标注采集时间。
- `错误原因.png`：具体失败项、置信区间、原图、前后回答和数据审核证据。
- `训练曲线.png`：两组真实CE/DPO曲线、完整内部指标；拟合与泛化分开。
- `latest.json`、`SNAPSHOT.json`：快照内容和生成状态。

PNG默认用已有Pillow与`/opt/data/private/.tools/fonts/NotoSansCJKsc-Regular.otf`离线生成，与HTML使用相同状态数据，不冒充浏览器截图。HTML交互已做JavaScript语法与HTTP接口检查，未执行浏览器点击自动化。可选浏览器渲染分支独立于训练环境，默认不依赖它。

本地实时服务：

```bash
source /opt/data/private/.tools/activate.sh
source /opt/data/private/.tools/vlm-a100-env.sh
"$VLM_A100_PYTHON" /opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001/tools/rarevlm_monitor.py --serve --port 8765
```

仅监听`127.0.0.1`；浏览器打开期间每15秒请求最新状态。`SERVER.json`记录实际URL、PID和启动时间。远程容器使用现有本地端口转发；文件快照与PNG不依赖服务，超过45秒会提示核对采集时间。端口被占用不抢占或终止原服务，可指定其他端口。容器更换后重启只读服务即可，训练恢复仍由原实验协议管理。

状态解释：

- 工程状态与指标结论分开。训练exit0可以同时“指标未达标”。
- 有效COMPLETION优先于旧队列状态；已独立恢复的历史KeyError明确标为已恢复。
- GPU未采集不等于空闲；未生成指标不填0；未通过门槛的阶段显示“未放行”。
- 损失曲线从逐QA trace增量汇总，仅展示完整16-QA更新；不把曲线下降当能力提升。
- 置信区间读取原分析，不重复bootstrap。点保护失败且区间跨0，不称已证明显著退化。
- 评测分数变化病例保留原标签/回答/评分，不自动当作新的训练样本。`case_notes.json`仅作事后解释，绑定研究名、角色、qid和前后答案，避免错误套用。

当前采集适配器对应`vlm_native_correction_sft_dpo_v2`的角色和数据格式。新实验需要登记路径并适配角色/阶段/指标，不能把未接入监控误判为没有运行。兼容该格式的研究可用`--run-root`指定根目录。源码：`tools/rarevlm_monitor.py`、`tools/rarevlm_monitor_raster.py`、`docs/monitor/template.html`。
