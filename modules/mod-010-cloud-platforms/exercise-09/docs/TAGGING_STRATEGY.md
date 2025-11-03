# AWS Tagging Strategy for Cost Allocation

## Executive Summary

**Purpose:** Implement comprehensive resource tagging to enable cost allocation, chargeback, and resource management across the ML infrastructure platform.

**Current State:**
- Only 60% of resources are tagged
- 40% of costs ($32K/month) cannot be attributed to teams
- No tag enforcement mechanisms
- Inconsistent tag naming conventions

**Target State:**
- 95%+ resources tagged within 60 days
- Full cost allocation and chargeback by team
- Automated tag enforcement
- Consistent tagging standards across all services

**Impact:**
- Enable accurate chargeback and budgeting
- Improve cost visibility and accountability
- Support cost optimization initiatives
- Facilitate compliance and audit requirements

---

## Tag Taxonomy

### Required Tags (Mandatory for All Resources)

These tags MUST be applied to all resources. Resources without these tags will trigger alerts and may be subject to automatic shutdown.

#### 1. `environment`

**Purpose:** Identify the environment/lifecycle stage

**Allowed Values:**
- `production` - Production workloads
- `staging` - Staging/pre-production environment
- `development` - Development environment
- `sandbox` - Temporary experimental resources
- `dr` - Disaster recovery resources

**Cost Impact:** Different cost optimization strategies per environment
- Production: Reserved Instances, high availability
- Staging: Auto-shutdown nights/weekends
- Development: Auto-shutdown daily, Spot instances
- Sandbox: 7-day auto-termination

**Example:**
```
environment=production
environment=development
```

#### 2. `team`

**Purpose:** Identify the team responsible for the resource

**Allowed Values:**
- `ml-platform` - ML Platform team
- `data-science` - Data Science team
- `ml-research` - ML Research team
- `ml-ops` - ML Operations team
- `data-engineering` - Data Engineering team
- `infrastructure` - Infrastructure team

**Cost Impact:** Enables team-level chargeback and budget allocation

**Example:**
```
team=ml-platform
team=data-science
```

#### 3. `project`

**Purpose:** Identify the project or product the resource supports

**Format:** `{project-name}` (lowercase, hyphens)

**Examples:**
- `fraud-detection`
- `recommendation-engine`
- `customer-churn`
- `image-classification`
- `feature-store`
- `model-registry`

**Cost Impact:** Enables project-level P&L tracking

**Example:**
```
project=fraud-detection
project=recommendation-engine
```

#### 4. `cost-center`

**Purpose:** Identify the cost center for chargeback

**Format:** `{dept}-{number}` (4-digit number)

**Examples:**
- `ml-1001` - ML Platform cost center
- `ds-2001` - Data Science cost center
- `mlr-3001` - ML Research cost center
- `eng-4001` - Engineering cost center

**Cost Impact:** Direct mapping to finance chargeback system

**Example:**
```
cost-center=ml-1001
cost-center=ds-2001
```

#### 5. `owner`

**Purpose:** Identify the individual responsible for the resource

**Format:** `{email}` (company email address)

**Examples:**
- `john.doe@company.com`
- `jane.smith@company.com`

**Cost Impact:** Accountability for orphaned resources and cost spikes

**Example:**
```
owner=john.doe@company.com
```

#### 6. `created-by`

**Purpose:** Track who/what created the resource

**Format:** `{user|service}`

**Examples:**
- `terraform` - Created by Terraform
- `john.doe@company.com` - Manually created
- `ci-cd` - Created by CI/CD pipeline
- `auto-scaling` - Created by auto-scaling

**Cost Impact:** Identify automation vs manual provisioning

**Example:**
```
created-by=terraform
created-by=john.doe@company.com
```

### Optional Tags (Recommended)

These tags provide additional context but are not mandatory.

#### 7. `service-type`

**Purpose:** Classify the type of service

**Examples:**
- `ml-training`
- `ml-inference`
- `database`
- `cache`
- `storage`
- `networking`

**Example:**
```
service-type=ml-training
service-type=ml-inference
```

#### 8. `data-classification`

**Purpose:** Indicate data sensitivity level

**Allowed Values:**
- `public`
- `internal`
- `confidential`
- `restricted`

**Example:**
```
data-classification=confidential
```

#### 9. `backup-policy`

**Purpose:** Indicate backup requirements

**Allowed Values:**
- `daily`
- `weekly`
- `none`
- `continuous`

**Example:**
```
backup-policy=daily
```

#### 10. `auto-shutdown`

**Purpose:** Control automated shutdown behavior

**Allowed Values:**
- `enabled` - Allow auto-shutdown
- `disabled` - Never auto-shutdown
- `nights` - Shutdown 8pm-8am
- `weekends` - Shutdown Sat-Sun
- `nights-weekends` - Shutdown nights and weekends

**Cost Impact:** Can save 50%+ on dev/staging costs

**Example:**
```
auto-shutdown=nights-weekends
auto-shutdown=disabled
```

#### 11. `expiry-date`

**Purpose:** Indicate when resource should be terminated

**Format:** `YYYY-MM-DD`

**Examples:**
- `2024-12-31`
- `2024-08-15`

**Cost Impact:** Automated cleanup of temporary resources

**Example:**
```
expiry-date=2024-12-31
```

---

## Tag Enforcement

### Enforcement Mechanisms

#### 1. AWS Config Rules

**Rule: required-tags**

Checks that resources have all required tags.

**Configuration:**
```json
{
  "ConfigRuleName": "required-tags",
  "Description": "Checks that resources have required tags",
  "Scope": {
    "ComplianceResourceTypes": [
      "AWS::EC2::Instance",
      "AWS::EC2::Volume",
      "AWS::RDS::DBInstance",
      "AWS::S3::Bucket",
      "AWS::SageMaker::NotebookInstance",
      "AWS::SageMaker::Endpoint"
    ]
  },
  "Source": {
    "Owner": "AWS",
    "SourceIdentifier": "REQUIRED_TAGS"
  },
  "InputParameters": {
    "tag1Key": "environment",
    "tag2Key": "team",
    "tag3Key": "project",
    "tag4Key": "cost-center",
    "tag5Key": "owner",
    "tag6Key": "created-by"
  }
}
```

**Actions on Non-Compliance:**
- Send SNS notification to owner
- Create Jira ticket for remediation
- After 7 days: Stop/terminate resource (except production)

#### 2. Service Control Policies (SCPs)

**Policy: Deny resource creation without required tags**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyEC2WithoutTags",
      "Effect": "Deny",
      "Action": [
        "ec2:RunInstances"
      ],
      "Resource": [
        "arn:aws:ec2:*:*:instance/*"
      ],
      "Condition": {
        "StringNotLike": {
          "aws:RequestTag/environment": "*",
          "aws:RequestTag/team": "*",
          "aws:RequestTag/project": "*",
          "aws:RequestTag/cost-center": "*",
          "aws:RequestTag/owner": "*"
        }
      }
    }
  ]
}
```

**Note:** This blocks creation at the API level - most restrictive approach.

#### 3. Terraform Validation

**Custom Policy:** Validate tags in Terraform plans

```hcl
# sentinel.hcl
policy "enforce-required-tags" {
  enforcement_level = "hard-mandatory"
}

# enforce-required-tags.sentinel
import "tfplan/v2" as tfplan

required_tags = ["environment", "team", "project", "cost-center", "owner", "created-by"]

main = rule {
  all tfplan.resource_changes as _, rc {
    all required_tags as tag {
      rc.change.after.tags[tag] exists
    }
  }
}
```

**Result:** Terraform apply will fail if required tags are missing.

#### 4. Lambda Auto-Remediation

**Function: auto-tag-resources**

Automatically applies default tags to new resources based on context.

```python
def lambda_handler(event, context):
    """
    Auto-tag newly created resources with default tags.
    Triggered by CloudWatch Events on resource creation.
    """
    resource_arn = event['detail']['resource-arn']
    creator = event['detail']['userIdentity']['principalId']

    # Extract team from IAM role/user
    team = extract_team_from_iam(creator)

    # Apply default tags
    tags = {
        'created-by': creator,
        'team': team,
        'created-date': datetime.now().isoformat(),
        'auto-tagged': 'true'
    }

    apply_tags(resource_arn, tags)
```

### Compliance Monitoring

**Daily Report:**
- % of resources tagged
- Non-compliant resources by team
- Trend over time

**Weekly Review:**
- Review non-compliant resources with team leads
- Update tagging policies if needed
- Celebrate teams with 95%+ compliance

**Monthly Audit:**
- Full compliance audit
- Cost allocation accuracy check
- Tag value consistency review

---

## Chargeback System

### Cost Allocation Model

**1. Direct Costs**

Costs directly attributable to a team/project based on tags.

**Allocation:**
```
Team Cost = Sum of all resources tagged with team={team-name}
Project Cost = Sum of all resources tagged with project={project-name}
```

**Example:**
```
ML Platform Team = $26,400
  - fraud-detection project: $12,000
  - recommendation-engine project: $8,500
  - feature-store project: $5,900
```

**2. Shared Costs**

Costs for shared resources (networking, monitoring, security).

**Allocation Methods:**

**a) Equal Split:**
```
Team Share = Shared Cost / Number of Teams
```

**b) Proportional Split (Recommended):**
```
Team Share = Shared Cost × (Team Direct Cost / Total Direct Cost)
```

**Example:**
```
Shared Costs = $10,000/month
ML Platform direct cost = $26,400 (40% of total)
ML Platform share of shared = $10,000 × 0.40 = $4,000
```

**3. Untagged Costs**

Costs for resources without proper tags.

**Allocation:**
- Month 1-2: Absorbed by central IT budget
- Month 3+: Split proportionally across all teams
- After 6 months: Charged to team with most untagged resources

**Incentive:** Strong motivation to maintain tagging compliance

### Chargeback Report Structure

**Monthly Chargeback Report (Per Team):**

```markdown
# ML Platform Team - Cost Report
## Period: June 2024

### Direct Costs by Service
| Service | Cost | % of Team Total |
|---------|------|-----------------|
| EC2 | $18,500 | 62% |
| SageMaker | $5,000 | 17% |
| S3 | $2,400 | 8% |
| RDS | $1,800 | 6% |
| Other | $2,000 | 7% |
| **Total Direct** | **$29,700** | **100%** |

### Allocated Shared Costs
| Category | Cost | Allocation Method |
|----------|------|-------------------|
| Networking | $1,200 | Proportional (40%) |
| Monitoring | $800 | Proportional (40%) |
| Security | $600 | Proportional (40%) |
| **Total Shared** | **$2,600** | |

### Untagged Resources
| Resource Type | Count | Cost |
|---------------|-------|------|
| EC2 instances | 3 | $450 |
| EBS volumes | 8 | $120 |
| **Total Untagged** | **11** | **$570** |

### Total Team Cost: $32,870

### Cost by Project
| Project | Cost | % of Team |
|---------|------|-----------|
| fraud-detection | $12,000 | 37% |
| recommendation-engine | $8,500 | 26% |
| feature-store | $5,900 | 18% |
| model-registry | $3,300 | 10% |
| Unallocated | $3,170 | 9% |

### Month-over-Month Trend
- Previous month: $30,200
- Current month: $32,870
- Change: +$2,670 (+8.8%)

### Top Cost Drivers
1. fraud-detection EC2 instances: +$1,800
2. New SageMaker endpoints: +$1,200
3. Increased S3 storage: +$400

### Recommendations
1. Right-size fraud-detection EC2 instances (over-provisioned)
2. Use SageMaker Savings Plans for endpoints
3. Implement S3 lifecycle policies
```

### Chargeback Automation

**Data Pipeline:**

```
AWS Cost Explorer API
    ↓
Extract costs by tags
    ↓
Allocate shared costs
    ↓
Generate team reports
    ↓
Send to finance system
    ↓
Email reports to team leads
```

**Implementation:**
- Daily: Extract cost data
- Weekly: Generate preliminary reports
- Monthly: Finalize and distribute chargeback reports

---

## Implementation Plan

### Phase 1: Foundation (Weeks 1-2)

**Objectives:**
- Define tag taxonomy
- Configure tag enforcement
- Create documentation

**Tasks:**
1. ✅ Define required and optional tags
2. ✅ Document tag standards
3. Create tag governance policy document
4. Set up AWS Config rules for tag enforcement
5. Create Lambda functions for auto-remediation
6. Configure SNS notifications for non-compliance

**Deliverables:**
- Tag taxonomy document (this document)
- AWS Config rules deployed
- Auto-remediation Lambda functions
- Tag compliance dashboard

**Success Criteria:**
- All new resources must have required tags
- Non-compliance notifications working
- Team acknowledgment of tagging standards

### Phase 2: Remediation (Weeks 3-6)

**Objectives:**
- Tag existing untagged resources
- Achieve 90%+ compliance
- Train teams on tagging

**Tasks:**
1. Inventory all untagged resources
2. Assign ownership for remediation
3. Bulk-tag resources by environment/team
4. Manual review for ambiguous resources
5. Weekly compliance review meetings
6. Train teams on tagging best practices

**Deliverables:**
- 90%+ resources tagged
- Weekly compliance reports
- Team training materials

**Success Criteria:**
- <10% untagged resources
- All teams understand tagging requirements
- Remediation process established

### Phase 3: Chargeback (Weeks 7-10)

**Objectives:**
- Implement cost allocation
- Generate chargeback reports
- Integrate with finance systems

**Tasks:**
1. Design cost allocation model
2. Build chargeback report pipeline
3. Generate pilot reports for 1-2 teams
4. Refine allocation model based on feedback
5. Integrate with finance/accounting systems
6. Distribute first official chargeback reports

**Deliverables:**
- Automated chargeback reports
- Finance system integration
- Monthly cost allocation by team/project

**Success Criteria:**
- 95%+ cost allocation accuracy
- Teams can validate their costs
- Finance accepts reports for budgeting

### Phase 4: Optimization (Weeks 11-12)

**Objectives:**
- Leverage tags for cost optimization
- Establish continuous improvement
- Measure ROI of tagging initiative

**Tasks:**
1. Implement auto-shutdown based on tags
2. Right-size instances by environment tags
3. Apply Reserved Instances to production tagged resources
4. Optimize storage based on project tags
5. Establish monthly tagging review process
6. Measure cost savings from tag-based optimization

**Deliverables:**
- Tag-based optimization automations
- Monthly governance review process
- ROI report on tagging initiative

**Success Criteria:**
- 10%+ cost reduction from tag-based optimization
- 95%+ sustained tagging compliance
- Zero untagged production resources

---

## Tag Management Tools

### 1. Tag Editor (AWS Console)

**Use Case:** Bulk tagging existing resources

**Steps:**
1. Navigate to Resource Groups → Tag Editor
2. Search for resources by type/region
3. Filter untagged resources
4. Apply tags in bulk
5. Review and confirm changes

**Limitations:**
- Manual process
- No validation
- Risk of inconsistent values

### 2. AWS CLI

**Use Case:** Scripted tagging for automation

**Examples:**

**Tag EC2 instances:**
```bash
aws ec2 create-tags \
  --resources i-1234567890abcdef0 \
  --tags \
    Key=environment,Value=production \
    Key=team,Value=ml-platform \
    Key=project,Value=fraud-detection
```

**Tag all untagged instances in a region:**
```bash
# Find untagged instances
aws ec2 describe-instances \
  --query 'Reservations[].Instances[?!not_null(Tags[?Key==`environment`])].InstanceId' \
  --output text

# Tag them
for instance_id in $(cat untagged_instances.txt); do
  aws ec2 create-tags \
    --resources $instance_id \
    --tags Key=environment,Value=development \
           Key=team,Value=unknown \
           Key=owner,Value=unassigned
done
```

### 3. Terraform

**Use Case:** Tag resources at creation time

**Example:**
```hcl
locals {
  common_tags = {
    environment  = var.environment
    team         = "ml-platform"
    project      = "fraud-detection"
    cost-center  = "ml-1001"
    owner        = "john.doe@company.com"
    created-by   = "terraform"
  }
}

resource "aws_instance" "ml_training" {
  ami           = var.ami_id
  instance_type = "p3.2xlarge"

  tags = merge(
    local.common_tags,
    {
      Name         = "ml-training-server"
      service-type = "ml-training"
    }
  )
}
```

### 4. Custom Tagging Script

**Use Case:** Intelligent bulk tagging with validation

```python
#!/usr/bin/env python3
"""
Bulk tag AWS resources with validation.

Usage:
    python tag_resources.py --tag-file tags.json --dry-run
    python tag_resources.py --tag-file tags.json --apply
"""

import boto3
import json
import argparse
from typing import Dict, List

REQUIRED_TAGS = ['environment', 'team', 'project', 'cost-center', 'owner', 'created-by']

def validate_tags(tags: Dict[str, str]) -> List[str]:
    """Validate that all required tags are present."""
    missing = []
    for tag in REQUIRED_TAGS:
        if tag not in tags:
            missing.append(tag)
    return missing

def tag_resource(resource_arn: str, tags: Dict[str, str], dry_run: bool = False):
    """Apply tags to a resource."""
    # Validate tags
    missing = validate_tags(tags)
    if missing:
        print(f"ERROR: Missing required tags: {missing}")
        return False

    if dry_run:
        print(f"[DRY RUN] Would tag {resource_arn} with: {tags}")
        return True

    # Apply tags based on resource type
    if ':ec2:' in resource_arn:
        ec2 = boto3.client('ec2')
        resource_id = resource_arn.split('/')[-1]
        ec2.create_tags(
            Resources=[resource_id],
            Tags=[{'Key': k, 'Value': v} for k, v in tags.items()]
        )
    elif ':s3:' in resource_arn:
        s3 = boto3.client('s3')
        bucket_name = resource_arn.split(':')[-1]
        s3.put_bucket_tagging(
            Bucket=bucket_name,
            Tagging={'TagSet': [{'Key': k, 'Value': v} for k, v in tags.items()]}
        )
    # ... handle other resource types

    print(f"✓ Tagged {resource_arn}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Bulk tag AWS resources')
    parser.add_argument('--tag-file', required=True, help='JSON file with resource → tags mapping')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    parser.add_argument('--apply', action='store_true', help='Apply tags')

    args = parser.parse_args()

    # Load tag mappings
    with open(args.tag_file) as f:
        tag_mappings = json.load(f)

    # Apply tags
    success_count = 0
    error_count = 0

    for resource_arn, tags in tag_mappings.items():
        try:
            if tag_resource(resource_arn, tags, dry_run=args.dry_run):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"ERROR tagging {resource_arn}: {e}")
            error_count += 1

    print()
    print(f"Tagged {success_count} resources")
    if error_count > 0:
        print(f"Failed to tag {error_count} resources")

if __name__ == '__main__':
    main()
```

---

## Tag Governance

### Roles and Responsibilities

**Cloud FinOps Team:**
- Define and maintain tag taxonomy
- Monitor tagging compliance
- Generate chargeback reports
- Provide tagging guidance and support

**Team Leads:**
- Ensure team resources are properly tagged
- Review monthly compliance reports
- Remediate untagged resources
- Approve cost allocations

**Engineers/Data Scientists:**
- Tag all resources at creation time
- Follow tagging standards
- Update tags when ownership/project changes
- Report tagging issues

**Finance Team:**
- Validate cost allocation accuracy
- Provide budgets by team/project
- Review chargeback reports
- Integrate with accounting systems

### Tag Review Process

**Weekly:**
- Review tag compliance dashboard
- Identify newly non-compliant resources
- Send notifications to owners

**Monthly:**
- Full tagging audit
- Review chargeback accuracy
- Update tag taxonomy if needed
- Celebrate high-compliance teams

**Quarterly:**
- Review tag governance policy
- Assess ROI of tagging initiative
- Update enforcement rules
- Train new team members

### Tag Change Process

**Modifying Tag Values:**
1. Update resource tags via AWS Console, CLI, or Terraform
2. Document reason for change
3. Notify FinOps team if cost-center or team changes

**Adding New Tag Keys:**
1. Propose new tag to FinOps team
2. Document use case and benefit
3. Update tag taxonomy
4. Communicate to all teams
5. Update enforcement rules

**Deprecating Tag Keys:**
1. Announce deprecation with 90-day notice
2. Identify resources using deprecated tag
3. Migrate to new tag
4. Remove from enforcement rules

---

## Metrics and KPIs

### Tagging Compliance Metrics

**1. Tag Coverage Rate**
```
Tag Coverage = (Resources with all required tags / Total resources) × 100%
```
**Target:** 95%

**2. New Resource Compliance**
```
New Resource Compliance = (New resources tagged / Total new resources) × 100%
```
**Target:** 99%

**3. Compliance by Team**
```
Team Compliance = (Team's tagged resources / Team's total resources) × 100%
```
**Target:** 95% per team

**4. Tag Value Consistency**
```
Consistency = (Tags with valid values / Total tag instances) × 100%
```
**Target:** 98%

### Cost Allocation Metrics

**1. Allocated Cost Percentage**
```
Allocated % = (Costs with full tags / Total costs) × 100%
```
**Target:** 95%

**2. Unallocated Cost**
```
Unallocated Cost = Total Monthly Cost - Allocated Cost
```
**Target:** <$5,000/month (<5% of total)

**3. Chargeback Accuracy**
```
Accuracy = (Validated charges / Total charges) × 100%
```
**Target:** 98% (based on team validation)

### Optimization Metrics

**1. Cost Savings from Tag-Based Actions**
- Auto-shutdown savings
- Right-sizing savings
- Reserved Instance matching

**Target:** $10,000+/month

**2. Time to Remediate Non-Compliance**
```
Remediation Time = Days from detection to tagging
```
**Target:** <7 days

**3. Orphaned Resource Detection Rate**
```
Detection Rate = (Orphaned resources found / Total resources) × 100%
```
**Target:** Detect 100% within 30 days of becoming orphaned

---

## Common Tagging Scenarios

### Scenario 1: New ML Training Job

**Context:** Data scientist launches SageMaker training job

**Required Tags:**
```python
tags = {
    'environment': 'production',
    'team': 'data-science',
    'project': 'fraud-detection',
    'cost-center': 'ds-2001',
    'owner': 'jane.smith@company.com',
    'created-by': 'terraform',
    'service-type': 'ml-training',
    'auto-shutdown': 'disabled'  # Training jobs terminate automatically
}
```

### Scenario 2: Temporary Dev Instance

**Context:** Engineer needs temporary EC2 for debugging

**Required Tags:**
```python
tags = {
    'environment': 'sandbox',
    'team': 'ml-platform',
    'project': 'debugging',
    'cost-center': 'ml-1001',
    'owner': 'john.doe@company.com',
    'created-by': 'john.doe@company.com',
    'service-type': 'compute',
    'auto-shutdown': 'nights-weekends',
    'expiry-date': '2024-08-15'  # Auto-terminate in 7 days
}
```

### Scenario 3: Shared Infrastructure

**Context:** EKS cluster used by multiple teams

**Required Tags:**
```python
tags = {
    'environment': 'production',
    'team': 'infrastructure',  # Owned by infra team
    'project': 'shared-kubernetes',
    'cost-center': 'eng-4001',  # Costs will be split across teams
    'owner': 'infra-team@company.com',
    'created-by': 'terraform',
    'service-type': 'kubernetes'
}
```

**Note:** Costs allocated proportionally to teams based on namespace usage

### Scenario 4: CI/CD Pipeline Resources

**Context:** Jenkins agents created by CI/CD

**Required Tags:**
```python
tags = {
    'environment': 'production',
    'team': 'ml-ops',
    'project': 'ci-cd-infrastructure',
    'cost-center': 'ml-1001',
    'owner': 'ci-cd@company.com',
    'created-by': 'ci-cd',
    'service-type': 'build-agent',
    'auto-shutdown': 'enabled'  # Terminate when builds complete
}
```

---

## Troubleshooting

### Issue 1: "Resource created without tags"

**Symptom:** AWS Config rule detects non-compliant resource

**Root Causes:**
1. Created via Console without tags
2. Auto-scaled resource (ASG/EKS)
3. Created by third-party tool

**Solutions:**
1. Use Tag Editor to add tags retroactively
2. Configure launch templates with tags for ASG
3. Update third-party tool configuration

**Prevention:**
- Use Terraform/CloudFormation for infrastructure
- Configure default tags in launch templates
- Enable SCP to block creation without tags

### Issue 2: "Inconsistent tag values"

**Symptom:** Same concept tagged differently (e.g., "prod" vs "production")

**Root Causes:**
1. No tag value validation
2. Manual tagging
3. Multiple Terraform workspaces with different values

**Solutions:**
1. Standardize on one value (e.g., "production")
2. Update AWS Config rule to validate values
3. Bulk-update incorrect values

**Prevention:**
- Use AWS Config rules with allowed values
- Terraform variables with validation
- Documentation and training

### Issue 3: "Cost allocation doesn't add up"

**Symptom:** Team costs don't match AWS bill

**Root Causes:**
1. Untagged resources
2. Shared costs not properly allocated
3. Data transfer costs not attributed

**Solutions:**
1. Tag all resources (target 95%+)
2. Review shared cost allocation model
3. Allocate data transfer based on source/destination tags

**Prevention:**
- Daily compliance monitoring
- Monthly allocation accuracy reviews
- Clear documentation of allocation model

### Issue 4: "Tags removed or changed unexpectedly"

**Symptom:** Resource tags disappear or change

**Root Causes:**
1. Terraform state drift
2. Manual changes via Console
3. Auto-scaling group replacement

**Solutions:**
1. Run `terraform apply` to restore tags
2. Use AWS Config auto-remediation
3. Ensure ASG launch template has correct tags

**Prevention:**
- Terraform state management
- Restrict Console access (read-only)
- Tag ASG launch templates

---

## Appendix

### A. Tag Value Reference

**Environment Values:**
- `production` - Live production systems
- `staging` - Pre-production staging
- `development` - Active development
- `sandbox` - Temporary experimentation
- `dr` - Disaster recovery

**Team Values:**
- `ml-platform` - ML Platform Engineering
- `data-science` - Data Science
- `ml-research` - ML Research
- `ml-ops` - ML Operations
- `data-engineering` - Data Engineering
- `infrastructure` - Infrastructure/DevOps

**Service Type Values:**
- `ml-training` - Model training
- `ml-inference` - Model serving/inference
- `database` - Database services
- `cache` - Caching layer
- `storage` - Object/file storage
- `networking` - Network infrastructure
- `monitoring` - Observability
- `security` - Security services

### B. AWS Config Rule Examples

**Required Tags Rule:**
```json
{
  "ConfigRuleName": "required-tags",
  "Description": "Check required tags exist",
  "Scope": {
    "ComplianceResourceTypes": [
      "AWS::EC2::Instance",
      "AWS::EC2::Volume",
      "AWS::RDS::DBInstance",
      "AWS::S3::Bucket"
    ]
  },
  "Source": {
    "Owner": "AWS",
    "SourceIdentifier": "REQUIRED_TAGS"
  },
  "InputParameters": "{\"tag1Key\":\"environment\",\"tag2Key\":\"team\",\"tag3Key\":\"project\",\"tag4Key\":\"cost-center\",\"tag5Key\":\"owner\",\"tag6Key\":\"created-by\"}"
}
```

**Valid Tag Values Rule:**
```json
{
  "ConfigRuleName": "environment-tag-valid-values",
  "Description": "Check environment tag has valid value",
  "Scope": {
    "ComplianceResourceTypes": ["AWS::EC2::Instance"]
  },
  "Source": {
    "Owner": "AWS",
    "SourceIdentifier": "REQUIRED_TAGS"
  },
  "InputParameters": "{\"tag1Key\":\"environment\",\"tag1Value\":\"production,staging,development,sandbox,dr\"}"
}
```

### C. Lambda Auto-Remediation Example

```python
import boto3
import json
from datetime import datetime

def lambda_handler(event, context):
    """
    Auto-remediate non-compliant resources.
    Triggered by AWS Config compliance change.
    """
    config = boto3.client('config')

    # Get non-compliant resources
    response = config.describe_compliance_by_config_rule(
        ConfigRuleNames=['required-tags'],
        ComplianceTypes=['NON_COMPLIANT']
    )

    for rule in response['ComplianceByConfigRules']:
        # Get resource details
        resources = config.get_compliance_details_by_config_rule(
            ConfigRuleName=rule['ConfigRuleName'],
            ComplianceTypes=['NON_COMPLIANT']
        )

        for resource in resources['EvaluationResults']:
            resource_id = resource['EvaluationResultIdentifier']['EvaluationResultQualifier']['ResourceId']
            resource_type = resource['EvaluationResultIdentifier']['EvaluationResultQualifier']['ResourceType']

            # Apply default tags
            if resource_type == 'AWS::EC2::Instance':
                apply_default_ec2_tags(resource_id)
            elif resource_type == 'AWS::S3::Bucket':
                apply_default_s3_tags(resource_id)

            print(f"Auto-remediated {resource_type} {resource_id}")

    return {'statusCode': 200, 'body': 'Remediation complete'}

def apply_default_ec2_tags(instance_id):
    """Apply default tags to EC2 instance."""
    ec2 = boto3.client('ec2')

    default_tags = {
        'environment': 'unclassified',
        'team': 'unknown',
        'owner': 'unassigned',
        'created-by': 'auto-remediation',
        'auto-tagged': 'true',
        'requires-review': 'true'
    }

    ec2.create_tags(
        Resources=[instance_id],
        Tags=[{'Key': k, 'Value': v} for k, v in default_tags.items()]
    )
```

---

**Document Version:** 1.0
**Last Updated:** 2024-06-30
**Next Review:** 2024-09-30
**Owner:** Cloud FinOps Team
