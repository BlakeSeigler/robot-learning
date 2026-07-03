# HPC / Slurm Quick-Start Guide

Personal notes on how to get onto a Slurm-managed HPC cluster, find GPUs, request
resources, and run/monitor training jobs. Written while setting up OpenVLA
finetuning. Institution-specific details (hostnames, account names, exact
partition names) are intentionally left as placeholders — fill in your own
cluster's values with the discovery commands below.

## Why any of this is necessary

OpenVLA-7B is a 7-billion-parameter vision-language-action model. That size is
the whole reason a laptop or a single consumer GPU can't do this:

- **Full finetuning** means updating every one of those 7B weights. At
  minimum you need the weights themselves in memory (~14GB in bf16), plus a
  gradient for every weight (another ~14GB), plus Adam's optimizer state
  (which keeps two extra running averages per weight — another ~56GB). That's
  well over 80GB before you've even loaded a training batch, which is why
  OpenVLA's own authors used 8x A100s for full finetuning — it doesn't fit on
  one GPU at all.
- **LoRA (Low-Rank Adaptation)** is the trick that makes this tractable on
  one GPU: freeze the base 7B model entirely, and only train a small set of
  extra low-rank "adapter" matrices bolted onto it. Gradients and optimizer
  state only need to be kept for those adapter weights, not the full 7B —
  which is what shrinks the memory footprint down to roughly 27GB for a
  rank-32 LoRA run at batch size 16 (per OpenVLA's own repo). That's why the
  LoRA config in the batch script below (`--lora_rank 32`, `--batch_size 16`)
  isn't arbitrary — those specific numbers are what's known to fit in that
  ~27GB envelope.

That ~27GB number is the entire reason the GPU-hunting below matters: it's
why a 40GB A100 or a 48GB L40S both work with headroom, while an 8–16GB
consumer card wouldn't. It's also *why* this needs an HPC at all rather than
a personal machine — GPUs with that much VRAM are expensive, scarce, shared
resources, and Slurm exists specifically to time-share that scarce hardware
across many researchers instead of everyone needing their own $10k+ card.
The partition/QOS/queueing machinery below is all just the mechanics of
"stand in line for a turn on one of these."

## Picking between GPU types when more than one would fit

Once you know how much VRAM your job actually needs (see above — ~27GB for
a rank-32 LoRA finetune of a 7B model), you'll often find *multiple* GPU
types on the cluster clear that bar. Example from a real cluster: both a
40GB Ampere-class card (e.g. A100) and a 48GB Ada Lovelace-class card (e.g.
L40S) satisfy the VRAM requirement equally. When that happens, VRAM isn't
actually the deciding factor anymore — these are:

**Higher memory-bandwidth cards (HBM2e-class, e.g. A100):**
- Faster per-step training throughput, all else equal — transformer training
  is largely memory-bandwidth bound, so this often means less wall-clock
  time once the job is running.
- More battle-tested for large-scale training specifically — most ML
  training libraries/kernels were developed and tuned against this class of
  card first, so it's the lowest-risk choice for hitting a weird
  compatibility bug.
- Downside: these pools tend to be smaller on shared clusters (more
  expensive, more contested), meaning **longer queue wait** before your job
  even starts.

**More VRAM, lower-bandwidth cards (GDDR6-class, e.g. L40S):**
- More headroom for larger batch sizes / higher LoRA rank if you want to
  push past a default config.
- Often far more numerous in a cluster's inventory (cheaper, more
  general-purpose cards) — can mean **much shorter queue time** in practice.
- Downside: somewhat slower per-step throughput than HBM-based cards; no
  NVLink (irrelevant for single-GPU jobs, matters only if scaling to
  multi-GPU later).

For a single-GPU job where required VRAM fits under both options, there
usually isn't a strong technical reason to prefer one — the practical
tiebreaker is just "which one has a shorter queue right now" (check with
`squeue -p <partition>`) traded off against "which one finishes faster once
it starts." Don't overthink this choice; either is a legitimate pick.

## Mental model

- **Cluster** — the whole HPC system.
- **Login node** — the machine you SSH into. No GPUs, minimal resources, only
  for submitting jobs / editing files / checking status. Never run real
  compute here.
- **Partition** — a named queue/grouping of nodes with similar hardware and
  access rules (e.g. an "A100 partition", an "L40S partition"). Just a label
  Slurm uses for scheduling, not a physical thing.
- **Node** — one physical server: multiple CPU cores, a chunk of system RAM,
  and (on GPU partitions) several physical GPU cards plugged into it.
- **Job** — your specific resource request. You get allocated a *slice* of a
  node (some CPUs, some RAM, some number of GPUs) — not the whole machine.
  Clusters are shared/multi-tenant: your job and several other users' jobs
  commonly run concurrently on the same physical node, each walled off from
  the others via cgroups.
- **Account / QOS** — separate from partitions. A partition defines what
  hardware exists; your account + QOS (quality-of-service) define whether
  *you* are allowed to use it and under what limits. You can see a valid
  partition and still get rejected if your QOS doesn't match what the
  partition requires.

## Step 0: find out what scheduler you're on

```bash
which sinfo squeue sbatch    # Slurm
which qstat pbsnodes qsub    # PBS/Torque
which bsub bhosts            # LSF
```

Everything below assumes Slurm.

## Discovering GPUs

```bash
# Partitions and their GPU resources (GRES = generic resource)
sinfo -o "%P %G %N"

# Show ALL partitions including hidden ones (sinfo hides some by default)
sinfo -a -o "%P %G %N"

# Just the GPU-having partitions
sinfo -o "%P %G" | grep -i gpu

# Per-node CPU/memory/state detail
sinfo -N -l

# Full config dump of a partition (time limits, allowed accounts/QOS, node list)
scontrol show partition <partition_name>

# Full detail on one specific node (look for the "Gres=" line)
scontrol show node <node_name>
```

`%G` output looks like `gpu:a100-pcie-40gb:3` or `gpu:l40s:8(S:0-1)` — GPU
type and count *per node*. The `(S:0-1)` means those GPUs are spread across
CPU sockets 0 and 1 on that node (physical NUMA/PCIe topology info — cores on
the "local" socket reach that GPU faster; not something to worry about for a
single-GPU job).

If `sinfo` shows GPU GRES types in `scontrol show config | grep -i gres` but
no partition actually lists GPU nodes, the GPU hardware may be on a
*different* cluster/login-node in a federated Slurm setup. Check:

```bash
scontrol show config | grep -i gres   # is GresTypes=gpu even configured?
sacctmgr show cluster                  # is this Slurm instance federated?
sinfo -M all -o "%P %G %N" 2>/dev/null # GPU partitions across all clusters
```

## Checking your account / QOS / partition access

Having a valid partition doesn't mean you can use it — check what your
account is actually authorized for:

```bash
sacctmgr show assoc user=$USER format=Account%30,Partition,QOS%40
```

Cross-reference against what a partition requires:

```bash
scontrol show partition <partition_name> | grep -i qos
# look for: AllowQos=gpu_access,gpu_access_plus
```

If your QOS list doesn't overlap with `AllowQos`, you'll get:
`srun: error: Unable to allocate resources: Invalid qos specification`

Fix by specifying the matching QOS (and account, if you have more than one)
explicitly:

```bash
srun --partition=<partition> --qos=<qos_name> --account=<account_name> ...
```

If `sacctmgr show assoc` shows only one account, `--account` is usually
optional (Slurm defaults to it). `--qos` commonly is *not* auto-selected
correctly and needs to be explicit.

## Storage — don't dump everything in $HOME

```bash
env | grep -i -E "scratch|project|work"   # cluster-set storage env vars
quota -s                                   # your home directory quota
df -h $HOME                                # home mount size/usage
```

Typical tiers on most HPCs:

- `$HOME` — small quota, backed up, slow. Scripts/configs/small code only.
- `$SCRATCH` / `/scratch/$USER` — large, fast, usually **not backed up**,
  often auto-purged after 30–90 days. Job I/O, datasets, checkpoints go here.
- `/project/<group>` or `/work/<group>` — shared, larger quota, longer-term.
- Node-local disk (set via `$TMPDIR` during a job) — fastest, exists only for
  the job's lifetime.

Datasets and model checkpoints for finetuning should live in scratch/work,
never `$HOME`.

## Interactive access (testing before committing to a batch job)

Quick one-off command (grabs a node, runs the command, releases automatically):

```bash
srun --partition=<gpu_partition> --qos=<qos> --gres=gpu:<gpu_type>:1 \
     --time=00:05:00 nvidia-smi
```

Full interactive shell (for poking around / debugging your env):

```bash
srun --partition=<gpu_partition> --qos=<qos> --gres=gpu:<gpu_type>:1 \
     --time=00:30:00 --pty bash
```

- `--gres=gpu:N` is what actually reserves N GPU(s) for your cgroup. Without
  it you land on a GPU-equipped node but have **zero GPU access** — CUDA
  sees nothing.
- `--pty` allocates a real pseudo-terminal so an interactive shell behaves
  normally (arrow keys, history, Ctrl-C). Skip it for one-off non-interactive
  commands.
- `srun` queues just like `sbatch` — if no matching GPU is free, your
  terminal blocks until one becomes available (`squeue -u $USER` in another
  window shows `PD` = pending).
- Requesting `gpu:1` on an 8-GPU node gives you exactly 1 GPU; the other 7
  remain available for other jobs on the same physical machine. Use
  `--exclusive` only if you genuinely need the whole node.

### Watching GPU usage on a job that's already running

You can't run a second command in a terminal that's blocked running your
training job. Attach a *second* shell to the **same existing allocation**
(doesn't request new resources, doesn't queue) from another terminal:

```bash
squeue -u $USER                        # find your JOBID
srun --jobid=<JOBID> --pty bash        # attaches inside the existing job
nvidia-smi                             # or: watch -n 2 nvidia-smi
```

Some clusters also allow direct `ssh <node_name>` while you have an active
job there.

## Batch jobs (the real, long-running way)

```bash
#!/bin/bash
#SBATCH --job-name=openvla-lora
#SBATCH --partition=<gpu_partition>
#SBATCH --qos=<qos_name>
#SBATCH --gres=gpu:<gpu_type>:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=<scratch_path>/logs/%x-%j.out
#SBATCH --error=<scratch_path>/logs/%x-%j.err

cd <scratch_path>/openvla   # NOT $HOME — datasets/checkpoints are large

uv run torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir <scratch_path>/openvla/datasets \
  --dataset_name <dataset_name> \
  --run_root_dir <scratch_path>/openvla/runs \
  --adapter_tmp_dir <scratch_path>/openvla/adapter_tmp \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --save_steps 5000
```

Flag notes:

- Only `--partition` and `--gres` are *required* to get a GPU; everything
  else falls back to partition defaults. But defaults for CPUs/time are
  often too small for real training (e.g. 1 CPU core will starve your
  dataloader; a short default time limit will get your job killed
  mid-training) — set `--cpus-per-task` and `--time` explicitly.
- `--mem` requests **system RAM**, completely separate from GPU VRAM
  (`--gres` controls that). A 40GB-VRAM GPU node can easily have hundreds of
  GB of system RAM — `--mem=64G` and a 40GB GPU are not in conflict.

Submit and monitor:

```bash
sbatch finetune.sbatch                  # submit — returns a job ID
squeue -u $USER                          # PD=pending, R=running
tail -f <scratch_path>/logs/<name>-<jobid>.out   # live log
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
scancel <jobid>                          # kill it
```

## torchrun / distributed training on Slurm

`torchrun` doesn't know Slurm exists — Slurm's only job is handing you a node
+ GPU allocation. Once inside that allocation, `torchrun` just runs like it
would on any regular multi-GPU machine.

- `--standalone` forces single-node mode and auto-picks a **random free
  port** on localhost for process coordination ("rendezvous").
- Without `--standalone`, defaults typically fall back to a **fixed port**
  (historically `29500`). On a shared cluster where multiple users' jobs can
  land on the same physical node, this risks a port collision between
  unrelated jobs. For single-node training, always use `--standalone`.
- `--nproc-per-node` = number of GPUs you're using on that node; must match
  your `--gres=gpu:N` count.
- Multi-node training (rare for a single finetune) is the one case where you
  drop `--standalone` and instead wire `torchrun` to Slurm-provided env vars
  (`$SLURM_JOB_NODELIST`, `$SLURM_PROCID`, etc.) via explicit
  `--rdzv-id`/`--rdzv-backend`/`--rdzv-endpoint` flags.

## Troubleshooting log (real mistakes made along the way)

- `srun: fatal: No command given to execute` — `--pty` needs a command after
  it, e.g. `--pty bash`.
- `srun: error: invalid partition specified` — double check partition name
  characters carefully; e.g. a lowercase `l` (as in "l40") is easy to
  mistype as the digit `1`.
- `nvidia-smi: command not found` — you're on the login node. It has no GPU
  driver. Only compute nodes inside a GPU allocation have it.
- `Unable to allocate resources: Invalid qos specification` — your default
  QOS doesn't match what the partition allows; pass `--qos` explicitly (see
  the accounts/QOS section above).

## Quick reference

```bash
sinfo -o "%P %G %N"                                   # GPU partitions overview
sacctmgr show assoc user=$USER format=Account%30,Partition,QOS%40  # your access
squeue -u $USER                                        # your jobs
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode  # job history/result
scancel <jobid>                                         # kill a job
srun --partition=<p> --qos=<q> --gres=gpu:1 --time=00:05:00 nvidia-smi  # quick GPU check
```
