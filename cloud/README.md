# M5 on cloud — multi-provider train + serve

Provision a beefy ephemeral training VM and a small persistent serving VM on
**AWS**, **Azure**, **GCP**, or **Hetzner**. Driven by Terraform; all four
providers share the same cloud-init bootstrap and artifact-transport scripts.

## TL;DR

```bash
# Recommended: Hetzner (5–10× cheaper than the hyperscalers for equivalent compute)
export HCLOUD_TOKEN='...'
cd cloud/terraform/hetzner
cp terraform.tfvars.example terraform.tfvars   # then edit
cd ../../..
make cloud-init  PROVIDER=hetzner
make cloud-up    PROVIDER=hetzner   # provisions train + serve
make cloud-output                   # prints public IPs and serve URL
make cloud-status                   # curl /healthz on the serve VM
make cloud-down  PROVIDER=hetzner   # tear it all down
```

## Architecture

```
┌──────────────────────────┐                 ┌──────────────────────────┐
│  TRAIN VM (ephemeral)    │                 │  SERVE VM (persistent)   │
│  ─────────────────────   │  artifact via   │  ─────────────────────   │
│  cloud-init/train.sh:    │  S3 / Blob /    │  cloud-init/serve.sh:    │
│    1. install uv         │  GCS / S3-compat│    1. install Docker     │
│    2. clone repo         │  ───────────►   │    2. clone repo         │
│    3. m5 prep + train    │                 │    3. pull artifact      │
│    4. push_artifact.sh   │                 │    4. docker compose up  │
│    5. poweroff (opt.)    │                 │    5. wait for /healthz  │
└──────────────────────────┘                 └──────────────────────────┘
                                                        │
                                                        ▼
                                                http://<serve>:8000/forecast
```

The **same cloud-init / systemd / artifact-transport scripts run on every
provider** — only the Terraform module differs. Add a fifth provider by writing
a new `cloud/terraform/<name>/{versions,variables,main,outputs}.tf` that
matches the other four.

## Prerequisites

| Tool | Where | Required for |
|---|---|---|
| `terraform` ≥ 1.5 | https://developer.hashicorp.com/terraform/install | all |
| `aws` CLI v2 | https://aws.amazon.com/cli/ | AWS provider; also used as the default S3 client on the VMs |
| `az` CLI | https://learn.microsoft.com/cli/azure/install-azure-cli | Azure provider |
| `gcloud` CLI | https://cloud.google.com/sdk/docs/install | GCP provider |
| `hcloud` CLI (optional) | https://github.com/hetznercloud/cli | Hetzner — Terraform handles everything; `hcloud` only needed if you also script outside TF |

You don't need every CLI — only the one for the provider you'll actually use.

## Per-provider setup

### Hetzner (recommended for cost)

**Costs:**
- Train: `ccx33` (8 dedicated AMD vCPU / 32 GB RAM) ≈ €0.06/h. A 35-min training run costs about **€0.04**.
- Serve: `cpx21` (3 vCPU / 4 GB RAM) ≈ €5/month always-on.
- Object storage: ~€5/month for 1 TB.

**Setup:**
1. Create a project at https://console.hetzner.cloud
2. Generate an **API token** (Project → Security → API tokens, Read/Write)
3. Create an **Object Storage bucket** in the same location (Project → Object Storage). Note the location (`fsn1`, `nbg1`, `hel1`).
4. Generate **S3 access keys** (Project → Object Storage → Access Keys)

```bash
export HCLOUD_TOKEN='hetzner-api-token'
cd cloud/terraform/hetzner
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars      # paste in the bucket name + S3 keys
```

### AWS

**Costs:**
- Train: `c7i.2xlarge` (8 vCPU / 16 GB) ≈ $0.36/h. A 35-min run costs about **$0.21**.
- Serve: `t3.medium` (2 vCPU / 4 GB) ≈ $30/month always-on.
- S3: cents/month for the artifact (a few hundred MB).

**Setup:** the standard AWS SDK chain works — `~/.aws/credentials`, `aws sso login`, env vars, or an IAM role.

```bash
aws sts get-caller-identity        # verify auth
cd cloud/terraform/aws
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars           # set artifact_bucket_name (must be globally unique)
```

The Terraform module creates an IAM instance profile so the EC2 VMs read/write
the bucket without baked-in credentials.

### Azure

**Costs:**
- Train: `Standard_D8s_v5` (8 vCPU / 32 GB) ≈ $0.38/h.
- Serve: `Standard_B2s` (2 vCPU / 4 GB) ≈ $30/month always-on.

**Setup:**
```bash
az login
az account set --subscription '<subscription-id>'
cd cloud/terraform/azure
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars           # set storage_account_name (3–24 chars, lowercase + digits)
```

VMs use system-assigned managed identities; no key management needed.

### GCP

**Costs:**
- Train: `n2-standard-8` (8 vCPU / 32 GB) ≈ $0.39/h.
- Serve: `e2-medium` (2 vCPU / 4 GB) ≈ $25/month always-on.

**Setup:**
```bash
gcloud auth application-default login
gcloud config set project '<project-id>'
cd cloud/terraform/gcp
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars           # set project_id, artifact_bucket_name
```

VMs use a dedicated service account that's bound to the bucket via IAM.

## Workflow

### One-shot training, then keep serving

```bash
make cloud-init  PROVIDER=hetzner
make cloud-up    PROVIDER=hetzner    # both VMs come up; train kicks off automatically
                                      # train finishes in ~35 min and powers itself off
                                      # serve waits for /latest/ in the bucket then starts
make cloud-status PROVIDER=hetzner   # curl http://<serve>:8000/healthz
```

### Rotate to a fresh model

```bash
# 1. Boot a new train VM (serve keeps running on the old model).
make cloud-train-up PROVIDER=hetzner

# 2. After the train VM powers off, restart serve so it pulls the new artifact.
make cloud-ssh-serve PROVIDER=hetzner
# on the serve VM:
sudo systemctl restart m5-bootstrap.service     # re-runs serve.sh → pulls /latest → docker compose up
```

### Just spin up serve (using an artifact you've already trained elsewhere)

```bash
# Push your local artifact up to the bucket first:
bash cloud/scripts/push_artifact.sh artifacts/models/lgbm/latest s3://<bucket>/m5/lgbm/latest

# Then bring up serve only:
make cloud-serve-up PROVIDER=hetzner
```

### Tear everything down

```bash
make cloud-down PROVIDER=hetzner       # destroys VMs + networking
                                       # bucket persists by default — destroy manually if you want
```

## Cost summary

| Provider | Train (per 35-min run) | Serve (always-on /mo) | Object storage |
|---|---|---|---|
| Hetzner    | **€0.04**     | **€5**          | €5/TB |
| AWS        | $0.21         | $30             | $0.023/GB |
| Azure      | $0.22         | $30             | $0.018/GB |
| GCP        | $0.23         | $25             | $0.020/GB |

Stop the train VM between runs (it self-shutdowns by default) and the running cost
is just the serve VM + object storage. Hetzner is **~6× cheaper** than the
hyperscalers at this scale.

## Validation (no cloud spend)

```bash
make cloud-validate-all          # runs terraform validate on all four modules
make cloud-fmt                   # runs terraform fmt -recursive
```

Both run locally and don't touch any cloud account.

## Files

```
cloud/
├── Makefile.cloud               # included by top-level Makefile (cloud-* targets)
├── cloud-init/
│   ├── _user_data.sh.tftpl      # shared user-data wrapper (rendered per VM by Terraform)
│   ├── train.sh                 # role=train bootstrap (apt → uv → make prep+train → push)
│   └── serve.sh                 # role=serve bootstrap (apt → docker → pull → docker compose up)
├── systemd/
│   ├── m5-bootstrap.service     # re-runs serve.sh on every reboot (idempotent)
│   └── m5-train.service         # oneshot train wrapper
├── scripts/
│   ├── push_artifact.sh         # s3 / az / gs dispatcher (used by train.sh)
│   └── pull_artifact.sh         # mirror of push_artifact.sh (used by serve.sh)
└── terraform/
    ├── hetzner/                 # hcloud_server + hcloud_firewall + hcloud_ssh_key
    ├── aws/                     # aws_instance + aws_s3_bucket + IAM instance profile
    ├── azure/                   # azurerm_linux_virtual_machine + storage account + RBAC
    └── gcp/                     # google_compute_instance + GCS bucket + service account
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `terraform: command not found` | `brew install terraform` (macOS) or follow https://developer.hashicorp.com/terraform/install |
| Train VM never powers off | SSH in, check `journalctl -u m5-train.service -f` and `tail -f /var/log/m5-train.log`. The service has a 4-hour timeout; if it's still running, the LightGBM fit is likely just slow. |
| Serve VM /healthz never comes up | SSH in, `docker compose logs --tail=100`. Most common cause: artifact never appeared at the expected `/latest/` prefix in the bucket. Confirm with `aws s3 ls s3://<bucket>/m5/lgbm/latest/`. |
| `403` from /forecast | You set `serve_api_key` but the client isn't sending `X-API-Key`. Set `M5_SERVE_API_KEY=""` (empty) to disable auth. |
| Want to keep the bucket but tear down the VMs | `make cloud-train-up create_serve=false` then `make cloud-down` won't work directly — instead, use `terraform destroy -target` on the VM resources only. |
| Hetzner Object Storage 403s | The S3 access key you generated doesn't match the bucket's location, or the endpoint doesn't match (`fsn1`/`nbg1`/`hel1`). Both must agree. |

## Safety rails

- All `.tfvars`, `.terraform/`, and `terraform.tfstate*` files are gitignored.
- Each Terraform module pins its provider version range.
- `cloud-down` is the canonical teardown — never `rm -rf` the state directory.
- `make cloud-up` is `apply -auto-approve` — for interactive review, use `make cloud-plan` first.
