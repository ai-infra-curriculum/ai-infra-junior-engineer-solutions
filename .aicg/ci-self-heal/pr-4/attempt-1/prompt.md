# Address CI failures on PR #4

## Goal

The PR you just opened failed CI. Fix the failures listed
below by editing files on the current branch. Do NOT regenerate
the content from scratch — make the minimal edit needed to
satisfy each failing check.

## Failed checks

### 1. `kubectl apply --dry-run=client` (failure)

- Details: <https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions/actions/runs/26611669345/job/78418717010>
- Annotations:
  - `.github:2` (warning): Node.js 20 actions are deprecated. The following actions are running on Node.js 20 and may not work as expected: actions/checkout@v4, azure/setup-kubectl@v4. Actions will be forced to run with Node.js 24 by default starting June 2nd, 2026. Node.js 20 will be removed from the runner on September 16th
  - `.github:64` (failure): Process completed with exit code 5.
  - `projects/project-02-kubernetes-serving/kubernetes/service.yaml:?` (failure): kubectl dry-run rejected manifest
  - `projects/project-02-kubernetes-serving/kubernetes/ingress.yaml:?` (failure): kubectl dry-run rejected manifest
  - `projects/project-02-kubernetes-serving/kubernetes/hpa.yaml:?` (failure): kubectl dry-run rejected manifest
  - `projects/project-02-kubernetes-serving/kubernetes/deployment.yaml:?` (failure): kubectl dry-run rejected manifest
  - `projects/project-02-kubernetes-serving/kubernetes/configmap.yaml:?` (failure): kubectl dry-run rejected manifest

### 2. `Markdown lint` (failure)

- Details: <https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions/actions/runs/26611669330/job/78418716982>
- Annotations:
  - `.github:2` (warning): Node.js 20 actions are deprecated. The following actions are running on Node.js 20 and may not work as expected: actions/checkout@v4, DavidAnson/markdownlint-cli2-action@v16. Actions will be forced to run with Node.js 24 by default starting June 2nd, 2026. Node.js 20 will be removed from the runner 
  - `modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:104` (failure): modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:104:1 MD004/ul-style Unordered list style [Expected: plus; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:102` (failure): modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:102:1 MD004/ul-style Unordered list style [Expected: plus; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:101` (failure): modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:101:1 MD004/ul-style Unordered list style [Expected: plus; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:99` (failure): modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:99:1 MD004/ul-style Unordered list style [Expected: plus; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:92` (failure): modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:92:1 MD004/ul-style Unordered list style [Expected: plus; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:91` (failure): modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:91:1 MD004/ul-style Unordered list style [Expected: plus; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:89` (failure): modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:89:1 MD004/ul-style Unordered list style [Expected: plus; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:87` (failure): modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:87:1 MD004/ul-style Unordered list style [Expected: plus; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:86` (failure): modules/mod-010-cloud-platforms/exercise-06-ml-infrastructure-aws/README.md:86:1 MD004/ul-style Unordered list style [Expected: plus; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `modules/mod-005-docker-containers/exercise-08-production-ml-deployment/README.md:91` (failure): modules/mod-005-docker-containers/exercise-08-production-ml-deployment/README.md:91:1 MD004/ul-style Unordered list style [Expected: dash; Actual: plus] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md

## Output contract

- Edit ONLY files inside this repo on the current branch.
- Preserve the existing structure; do not delete sections.
- Do NOT touch CURRICULUM.md, README.md, or VERSIONS.md.
- One atomic commit covering all fixes is fine.
