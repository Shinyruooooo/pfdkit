# PFD-kit 工作摘要（跨会话用）

## 关键路径

- 本地源码：`/home/shaolei/software/PFD_01013/pfd-kit`（GitHub: `Shinyruooooo/pfdkit`，master）
- 本地运行环境：`conda activate PFD_latest`（editable 安装指向上面源码）
- HPC 源码：`hyx:/home/hyx/soft/pfd-kit`（`ssh hyx` = 192.168.20.7 用户 hyx，脏工作树，不提交）
- HPC 环境激活：`/home/hyx/script/env/activate_deepmd_dpa4c.sh`（dp + lmp 都在里面）
- 生产任务目录：`~/project/GST_resistance_drift/1_model_train_pfd/s1_crystal_melt_liquid/`（`finetune.json`）
- 测试任务目录：`~/project/GST_resistance_drift/1_model_train_pfd/test_lmp_e2e/`
- 本地 k8s：minikube + argo（namespace `argo`），无 argo CLI，用 `kubectl -n argo get workflow <name>`

## 本次大改（均已 push）

1. **LAMMPS 探索**（`pfd/exploration/task/lmp_template_task_group.py`、`lmp_param_task_group.py`、`pfd/op/run_lmp.py`、`prep_model.py`）
   - 模板模式：用户自写 LAMMPS 脚本（`input_lmp_template` + `revisions` 覆盖 variable），契约：`pair_style deepmd pfd_model.pt2`、`dump pfd_traj` 含 fx/fy/fz。
   - 参数化模式：ASE 风格配置（ens/temps/press/dt/nsteps/tau_t/tau_p，dt/tau 单位 fs，press 单位 bar），自动生成 LAMMPS 输入。
   - LAMMPS 探索用冻结压缩模型（`.pt2`）：每轮训练后自动跑 `PrepModelFreeze`（freeze+compress），ASE 不需要此步。
2. **dpa4c 训练修复**（`pfd/op/train/dp.py`）：`pytorch_expt` 后端 + `--skip-neighbor-stat`，互斥参数处理。
3. **type_map 修复**（`pfd/entrypoint/submit.py`，commit `cef5ebf` + `1c76f06`）：
   - `exploration.config.type_map` 缺失时：模板模式从 `pair_coeff` 解析；参数化模式读 stage 的 `type_map`；再不行回退到 `init_confs[0]` 的化学符号（如 `['Ge','Te']`）。
   - `init_confs` 读取已提前到 LAMMPS 块之前（否则回退报 `UnboundLocalError`）。
   - 不修会导致 `blk-select-confs` 报 `KeyError: 'TYPE_0'`（`TrajRenderLammps.type_map=None`）。
4. 早前 ASE MD 稳定性：`MDStabilityError` 受控提前结束、`md_failed.extxyz/json` 诊断、`vol_tol`/`max_force` 可配置、局部 RNG。

## 操作注意事项

- **push**：本地 commit 后 `git push origin HEAD`（远程即 master）。
- **HPC 同步**：HPC 仓库脏，不能整库 rsync，只同步改动文件：
  `scp pfd/entrypoint/submit.py hyx:/home/hyx/soft/pfd-kit/pfd/entrypoint/submit.py`
  同步后用 `ssh hyx 'cd /home/hyx/soft/pfd-kit && cat <file>' | diff - <file>` 验证。
- **resubmit**（必须在任务目录下运行，路径是相对的）：
  `cd s1_crystal_melt_liquid && pfd resubmit finetune.json <wfid> -u 0-N [-m]`
  - `-u` 不传则直接返回不提交；`-l` 列出可复用步骤序号；`-m` = 不监控（action=store_false）。
  - **已知问题**：dflow reuse 对切片步骤不完全命中（4 个 run-expl 只复用了 3 个，最后一个会重跑）。复用前想清楚成本。
  - 长时监控建议 nohup 后台，别挂着前台进程（会重复提交）。
- **finetune.json 要点**：探索 `type: lmp`；运行命令/环境变量写在 `step_configs`（`run_explore_config`/`run_train_config` 的 `command`、`envs`：OMP/DP_INTRA/DP_INTER=6/6/1）；`train.config.impl: pytorch_expt`。
- **测试**：`pytest -q tests/test_lmp*.py`（29 passed）。pytest 只在 PFD_latest 环境有。
- **额度**：别整段拉 workflow YAML，用 `kubectl -o jsonpath` 取 `.status.phase` 等小字段。

## 当前状态

- `si-ft-local-jt55d`（从 `4znr2` resubmit）运行中，select-confs 已过（type_map 修复生效），进入后续 fp/train。
- 生产配置 `s1_crystal_melt_liquid/finetune.json`：GeTe，NPT 300/1000/1500/2000K × 1/10000 bar，nsteps 500000，max_iter 20。
