# Exercise 06: ML Infrastructure on AWS — Solution

## What the exercise asked for

Provision a minimal but realistic ML infrastructure on AWS:
S3 for artifacts, ECR for images, EKS for orchestration, IAM
for access, and the IaC (Terraform) that ties it together.

## Reference structure

See [`main.tf`](./main.tf) for the Terraform skeleton.

What it provisions:

1. **S3 buckets**: separate buckets for models, datasets, and
   audit logs (different lifecycle policies + encryption).
2. **ECR repository**: image registry with vulnerability
   scanning.
3. **EKS cluster** (skeleton): VPC, node groups, IAM roles for
   service accounts (IRSA).
4. **IAM**: per-workload identities mapped to specific S3
   prefixes (least privilege).
5. **KMS**: customer-managed keys for at-rest encryption.

## Operational walkthrough

```bash
# Init + plan
terraform init
terraform plan -out=plan.binary

# Apply (interactive — review the plan first)
terraform apply plan.binary

# Verify
aws eks list-clusters
aws s3 ls
aws ecr describe-repositories
```

## Key design decisions

### S3 bucket layout

```text
my-org-ml-models/<env>/<model-name>/<version>/
my-org-ml-datasets/<env>/<dataset>/<version>/
my-org-ml-audit/<env>/YYYY/MM/DD/
```

Per-purpose buckets enable per-purpose retention + encryption
+ access policies. A single bucket for everything is
operationally simpler at first; gets expensive in audit.

### IAM via IRSA (IAM Roles for Service Accounts)

```hcl
resource "aws_iam_role" "ml_serving" {
  name = "ml-serving-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.serving_trust.json
}

# Allow the pod's service account to read only its own model
data "aws_iam_policy_document" "serving_policy" {
  statement {
    actions = ["s3:GetObject"]
    resources = ["arn:aws:s3:::my-org-ml-models/${var.environment}/recs/*"]
  }
}
```

Per-pod IAM scope keeps blast radius small. A compromised
serving pod can read only its own model — not the training
data, not other tenants' artifacts.

### KMS hierarchy

A root KMS key per environment + per-purpose data keys. Audit
logs get their own key (immutable retention policy).

## Cost considerations

For SmartRecs scale (small team, modest training cadence),
expected monthly costs:

- **EKS control plane**: ~$73/month per cluster.
- **EC2 worker nodes**: depends on workload; budget $2k-$5k
  for a non-trivial mixed fleet.
- **S3**: cheap (cents per GB) unless you're storing many
  TB of training data.
- **ECR**: cheap.
- **KMS**: $1/month per CMK + per-request fees.

Watch out for: NAT Gateway costs ($45/month + data transfer
fees), CloudWatch log retention.

## What this exercise deliberately doesn't cover

- **Multi-region**: covered in `mod-010 exercise-04-cross-
  region-replication` and the Engineer track's mod-109.
- **Multi-cloud**: senior-engineer track lab `mod-205`.
- **Full IaC patterns**: Engineer track's `mod-109-
  infrastructure-as-code`.
- **Production-grade Helm + ArgoCD**: covered in
  `engineer-solutions/mod-109`.

## Common mistakes

- **Public S3 buckets** for model storage. Use bucket policies
  + IAM, not public ACL.
- **One IAM role for everything** — defeats least privilege.
- **No KMS** — relying on S3 default encryption with AWS-
  managed keys. Fine for non-regulated; insufficient when
  customers ask for customer-managed keys.
- **Terraform state in S3 without locking** — `terraform
  apply` from two engineers simultaneously corrupts state.
  Use DynamoDB lock.
- **No tagging strategy** — costs aren't attributable.

## Cross-references

- Exercise prompt:
  `ai-infra-junior-engineer-learning/lessons/mod-010-cloud-platforms/exercises/exercise-06-ml-infrastructure-aws.md`
- Engineer-track Terraform patterns:
  `engineer-solutions/mod-109-infrastructure-as-code`.
- Security track for IAM + KMS depth:
  `ai-infra-security-learning/lessons/mod-002-zero-trust-architecture/`
  and `mod-003-cryptography-for-ml/`.
