# Input Script Guide

<style>
  p {
    text-align: justify;
  }
</style>

## Basics
### Host and Nodes
#### Workflow Host
PFD-kit is built on the `dflow` package, which uses the Python API of `ARGO` workflows. While designed for cloud-based workflows with `Kubernetes`, a local "debug" mode is available for convenience. No cloud services are required for local execution.

To submit workflows to a remote server, specify the following configuration:

```json
"dflow_config": {
    "host": "http://address.of.the.host:port"
},
"dflow_s3_config": {
    "endpoint": "address.of.the.s3.server:port"
},
```
For the `Bohrium` platform, use:

```json
"bohrium_config": {
    "username": "your_username",
    "password": "your_password",
    "project_id": 123456,
    "_comment": "all"
},
```
The workflow will be hosted on `https://workflows.deepmodeling.com`, and progress can be tracked at `https://workflows.deepmodeling.com/workflows`.

#### Node Settings
The `step_configs` section defines computational resources for tasks like DFT calculations, model training, and MD exploration. For local execution, no configuration is needed, as all tasks run on the local machine.

For remote HPC nodes (e.g., `Slurm` systems), configure as follows:

```json
"step_configs": {
    "run_fp_config": {
        "template_config": {},
        "executor": {
            "type": "dispatcher",
            "host": "your_host",
            "username": "your_username",
            "password": "your_password",
            "port": 22,
            "private_key_file": null,
            "remote_root": "/remote_root",
            "queue_name": "queue",
            "machine_dict": {
                "remote_profile": {
                    "timeout": 600
                }
            },
            "resources_dict": {
                "source_list": ["path_to_source_file"],
                "module_list": ["remote_module"],
                "custom_flags": ["custom_commands"]
            }
        },
        "template_slice_config": {
            "group_size": 1,
            "pool_size": 1
        }
    }
}
```
For `Kubernetes` services (e.g., `Bohrium`), specify the container image and machine type:

```json
"step_configs": {
    "run_fp_config": {
        "template_config": {
            "image": "vasp_image_paths"
        },
        "continue_on_success_ratio": 0.9,
        "executor": {
            "type": "dispatcher",
            "image_pull_policy": "IfNotPresent",
            "machine_dict": {
                "batch_type": "Bohrium",
                "context_type": "Bohrium",
                "remote_profile": {
                    "input_data": {
                        "job_type": "container",
                        "platform": "ali",
                        "scass_type": "c32_m64_cpu"
                    }
                }
            }
        }
    }
}
```

## Fine-Tuning
The example `si_ft.json` file defines workflow tasks. Specify the task type (e.g., "finetune") and skip initial data generation and training if the `Domains_SemiCond` branch already provides sufficient accuracy:

```json
"task": {
    "type": "finetune",
    "max_iter": 5,
    "init_ft": false,
    "init_train": false
}
```
The `inputs` section includes essential parameters and model files. Prepare exploration systems with proper perturbation in advance:

```json
"inputs": {
    "base_model_path": "DPA2_medium_28_10M_beta4.pt",
    "init_confs": {
        "prefix": "./",
        "confs_paths": ["./pert_si32.extxyz"]
    },
    "init_fp_confs": {
        "prefix": "./",
        "confs_paths": []
    }
}
```
The `exploration` section defines structural configuration exploration. In this example, MD simulations at 1000 K under varying pressures generate new configurations:

```json
"exploration": {
    "type": "ase",
    "config": {
        "calculator": "dp",
        "head": "Domains_SemiCond"
    },
    "stages": [
        [
            {
                "conf_idx": [0],
                "n_sample": 1,
                "ens": "npt",
                "dt": 2,
                "nsteps": 2000,
                "temps": [1000],
                "press": [1, 1000, 10000],
                "trj_freq": 10,
                "tau_t": 100,
                "tau_p": 500,
                "compressibility": 5e-6,
                "seed": 12345,
                "vol_tol": 0.2,
                "max_force": 100.0
            }
        ]
    ]
}
```

Notes on the MD parameters:

- `tau_t` / `tau_p`: temperature / pressure coupling time, both in **fs**.
- `compressibility`: compressibility in **1/bar**. It is material-dependent and **required for NPT** — there is no default; an NPT stage without it raises an error.
- `seed`: optional random seed for velocity initialization. Set it for reproducible MD; omit it to keep non-deterministic behavior.
- `no_pbc`: optional, set to `true` to disable periodic boundary conditions.
- `vol_tol`: allowed relative cell volume change in the stability watchdog, defaults to `0.2` (±20%). Increase it (e.g. `0.5`) for simulations with large physical volume changes such as phase transitions, or set it to `null` to disable the volume check entirely.
- `max_force`: max-force threshold (in **eV/Å**) of the stability watchdog, defaults to `50.0`. The default is conservative; for high-temperature liquid exploration in active learning, `80`–`100` eV/Å is usually more appropriate. Set it to `null` to disable the force check entirely — not recommended unless you are confident about the short-range behavior of your model.
- Stability watchdog: when temperature exceeds 5000 K, cell volume changes by more than ±`vol_tol`, max force exceeds `max_force`, or NaN/inf appears in energy/forces/stress/positions, the MD is **stopped early in a controlled way**: the exploration task returns normally with the partial trajectory (`traj`) plus failure diagnostics (`md_failed.extxyz`, `md_failed.json`), instead of failing and being retried. The diagnostic JSON records the stop reason, step, temperature, energy, max force, volume ratio, and `requested_nsteps` / `completed_steps` / `early_stopped: true`. The partial trajectory is still used downstream (frame extraction/selection works with however many frames exist), since the pre-explosion frames and the final unstable structure are valuable for active learning.

For example, a high-temperature liquid GeTe NPT stage:

```json
{
    "ens": "npt",
    "temp": 1500,
    "press": 1,
    "dt": 0.5,
    "tau_t": 500,
    "tau_p": 2000,
    "compressibility": 5.0e-6,
    "vol_tol": 0.5,
    "max_force": 100.0,
    "seed": null
}
```

### LAMMPS exploration with user-written templates

Set `"type": "lmp"` to explore with LAMMPS instead of ASE. In this mode you write the LAMMPS input script yourself (ensemble, ramp/hold stages, fixes, computes — full control), and PFD only rewrites a small, well-defined set of things:

- the model file name in the `pair_style deepmd` line — the template must use the placeholder **`pfd_model.pt2`**; PFD provides the frozen+compressed model under this name at run time (a `prep-model` step freezes/compresses the checkpoint with `dp --pt-expt freeze/compress` between iterations);
- the structure file name in `read_data` — rewritten to `conf.lmp` (PFD converts the sampled configuration);
- the dump with id **`pfd_traj`** — its frequency and file name are rewritten; the dump must contain `id type element x y z fx fy fz` columns (forces are needed downstream);
- `variable NAME equal ...` lines listed in `revisions` are overridden;
- `atom_modify map yes` is inserted automatically if missing (required by graph-lowered `.pt2` models).

If the template misses any of the contract (`pair_style` with the placeholder, `read_data`, a `pfd_traj` dump with force columns), submission fails with an explicit error. The `type_map` (element order of the generated data file) is taken from the template's `pair_coeff` line by default.

Example stage configuration:

```json
"exploration": {
    "type": "lmp",
    "config": {"command": "lmp -k on g 1 -sf kk"},
    "stages": [
        [
            {
                "conf_idx": [0],
                "n_sample": 1,
                "input_lmp_template": "./explore/in.melt.lammps",
                "revisions": {"TSTOP": 2000, "NSTEPS": 500000, "SEED": 12345},
                "trj_freq": 50
            }
        ]
    ]
}
```

Template example (`./explore/in.melt.lammps`):

```
units           metal
dimension       3
boundary        p p p
atom_style      atomic
read_data       structure.lmp
pair_style      deepmd pfd_model.pt2
pair_coeff      * * Ge Te
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes
variable        T equal 1000.0
variable        NSTEPS equal 2000
variable        SEED equal 12345
timestep        0.001
thermo          50
thermo_style    custom step time temp pe ke etotal press vol density
thermo_modify   flush yes
dump            pfd_traj all custom 1000 out.lammpstrj id type element x y z fx fy fz
dump_modify     pfd_traj element Ge Te
dump_modify     pfd_traj sort id
velocity        all create ${T} ${SEED} mom yes rot yes dist gaussian
fix             melt all nvt temp ${T} ${T} 0.1
run             ${NSTEPS}
```

Notes for LAMMPS exploration:

- `config.command`: the LAMMPS command prefix (e.g. `lmp -k on g 1 -sf kk` for Kokkos GPU); `-in`, `-log` and `-screen` are added automatically if absent.
- `config.max_temp`: thermo temperature ceiling (default 5000 K). When the log shows temperatures above it, NaN/inf thermo values, or "Lost atoms", the task is **stopped early in a controlled way** — the partial trajectory plus `md_failed.extxyz`/`md_failed.json` diagnostics are returned normally and feed the exploration-stability convergence check, exactly like the ASE path.
- The LAMMPS run uses the **frozen+compressed** model produced by the `prep-model` step; make sure the compute environment (`source_list`) provides both `lmp` with the deepmd pair style and `dp` for freezing.

The `select_confs` node filters unphysical configurations and compresses data using entropy-based measures:

```json
"select_confs": {
    "max_sel": 60,
    "frame_filter": [
        {"type": "distance"}
    ],
    "h_filter": {
        "chunk_size": 5,
        "_comment": "entropy-based filter"
    }
}
```
The `fp` section defines DFT calculation settings, including VASP input and pseudopotential files:

```json
"fp": {
    "type": "vasp",
    "task_max": 50,
    "run_config": {
        "command": "mpirun -n 32 vasp_std"
    },
    "inputs_config": {
        "incar": "INCAR.fp",
        "pp_files": {
            "Si": "POTCAR"
        },
        "kspacing": 0.2
    }
}
```
The `train` section specifies the pretrained model and training configuration:

```json
"train": {
    "type": "dp",
    "config": {
        "impl": "pytorch",
        "head": "Domains_SemiCond"
    },
    "template_script": "train.json"
}
```
The `evaluate` section tests the model against a test dataset. Iterations continue until convergence or the maximum cycle limit is reached. Convergence is achieved when the force RMSE falls below 0.06 eV/Å:

```json
"evaluate": {
    "test_size": 0.3,
    "model": "dp",
    "head": "Domains_SemiCond",
    "_comment": "The percentage for test",
    "converge": {
        "type": "force_rmse",
        "RMSE": 0.06
    }
}
```

Optionally, convergence can additionally require **exploration stability**: an iteration then counts as converged only when the exploration MD slices of that iteration did not stop early (temperature/force/NaN blowups detected by the stability watchdog) beyond the configured tolerances. Add the `expl_stability` section to `evaluate`:

```json
"evaluate": {
    "test_size": 0.3,
    "model": "dp",
    "converge": {
        "type": "force_rmse",
        "RMSE": 0.06
    },
    "expl_stability": {
        "enabled": true,
        "max_failed_slices": 1,
        "max_lost_fraction": null,
        "ignored_reasons": ["volume"],
        "consecutive_clean_iters": 2
    }
}
```

- `enabled`: the mere presence of the section enables the check; set to `false` to disable it explicitly.
- `max_failed_slices`: number of early-stopped slices tolerated per iteration (default `0`).
- `max_lost_fraction`: tolerated fraction of lost MD steps, summed over counted early-stopped slices and divided by the total slice count (default `null`, i.e. not applied).
- `ignored_reasons`: stop-reason keywords that are reported but do not count against convergence. Defaults to `["volume"]`, because `vol_tol` triggers can be physically real (phase transitions, high-pressure compression). Note: the watchdog still stops such trajectories early regardless of this setting — to also tolerate the volume change during the MD itself, increase `vol_tol` or set it to `null` in the stage config.
- `consecutive_clean_iters`: number of consecutive iterations that must pass both the accuracy and the stability check before advancing to the next exploration stage (default `1`). Since MD survival is stochastic when `seed` is unset, a value of `2` protects against lucky passes.

When the stability check is not configured (no `expl_stability` section), convergence behaves exactly as before: test-set accuracy only.

## Distillation
The distillation script is similar to fine-tuning, with key differences. Change the `task/type` to `dist` and label new frames using the fine-tuned model:

```json
"task": {
    "type": "dist",
    "max_iter": 5
},
"inputs": {
    "base_model_path": "path_to_teacher_model",
    ...
},
"train": {
    "type": "dp",
    "config": {
        "impl": "pytorch"
    },
    "template_script": "./dist_train.json"
},
"fp": {
    "type": "ase",
    "run_config": {
        "model_style": "dp",
        "inputs_config": {
            "batch_size": 500
        }
    }
}
```
