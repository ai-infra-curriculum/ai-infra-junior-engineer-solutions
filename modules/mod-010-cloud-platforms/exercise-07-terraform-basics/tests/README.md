# Terraform Tests

This directory contains Terratest-based tests for the ML infrastructure Terraform configuration.

## Prerequisites

1. Go 1.19 or later
2. AWS credentials configured
3. Terraform installed

## Installation

Install required Go modules:

```bash
go mod init terraform-ml-infrastructure-test
go get github.com/gruntwork-io/terratest/modules/terraform
go get github.com/gruntwork-io/terratest/modules/aws
go get github.com/stretchr/testify/assert
```

Or create a `go.mod` file:

```go
module terraform-ml-infrastructure-test

go 1.19

require (
    github.com/gruntwork-io/terratest v0.43.0
    github.com/stretchr/testify v1.8.4
)
```

Then run:
```bash
go mod download
```

## Running Tests

### Run All Tests
```bash
go test -v -timeout 30m
```

### Run Specific Test
```bash
go test -v -timeout 30m -run TestTerraformValidation
```

### Run Tests in Parallel
```bash
go test -v -timeout 30m -parallel 3
```

## Test Descriptions

### TestTerraformMLInfrastructure
Complete end-to-end test that:
- Creates all infrastructure resources
- Validates VPC configuration
- Validates EC2 instances
- Validates S3 buckets
- Validates IAM roles
- Validates Security Groups
- Destroys all resources

**Duration**: ~15-20 minutes
**Cost**: ~$0.10 (uses t3.micro)

### TestTerraformOutputs
Lightweight test that validates:
- All expected outputs are defined
- Terraform syntax is correct

**Duration**: ~1 minute
**Cost**: $0 (no resources created)

### TestTerraformValidation
Validation test that checks:
- Terraform syntax
- Configuration validity
- Code formatting

**Duration**: ~1 minute
**Cost**: $0 (no resources created)

## Best Practices

1. **Always run validation tests first**:
   ```bash
   go test -v -run TestTerraformValidation
   ```

2. **Run integration tests in isolated AWS account**:
   - Use a separate AWS account for testing
   - Set up cost alerts

3. **Clean up resources**:
   - Tests automatically clean up with `defer terraform.Destroy`
   - If tests fail, manually check AWS console

4. **Use random regions**:
   - Tests use random regions to avoid conflicts
   - Helps distribute load across regions

5. **Set appropriate timeouts**:
   - Infrastructure tests need 30+ minutes
   - Use `-timeout` flag to adjust

## Troubleshooting

### Test Timeout
If tests timeout, increase the timeout:
```bash
go test -v -timeout 60m
```

### AWS Credentials Error
Ensure AWS credentials are configured:
```bash
aws sts get-caller-identity
```

### Resources Not Cleaned Up
If a test fails and resources aren't destroyed:
```bash
cd ../terraform
terraform destroy -auto-approve
```

### Rate Limiting
AWS API rate limits may cause test failures. Add retry logic or run tests sequentially:
```bash
go test -v -timeout 30m -parallel 1
```

## Example Output

```
=== RUN   TestTerraformMLInfrastructure
=== PAUSE TestTerraformMLInfrastructure
=== CONT  TestTerraformMLInfrastructure
TestTerraformMLInfrastructure 2024-01-01T12:00:00Z logger.go:66: Running terraform init...
TestTerraformMLInfrastructure 2024-01-01T12:00:05Z logger.go:66: Running terraform apply...
TestTerraformMLInfrastructure 2024-01-01T12:05:00Z logger.go:66: Apply complete!
=== RUN   TestTerraformMLInfrastructure/ValidateVPC
=== RUN   TestTerraformMLInfrastructure/ValidateEC2
=== RUN   TestTerraformMLInfrastructure/ValidateS3
=== RUN   TestTerraformMLInfrastructure/ValidateIAM
TestTerraformMLInfrastructure 2024-01-01T12:07:00Z logger.go:66: Running terraform destroy...
--- PASS: TestTerraformMLInfrastructure (420.35s)
    --- PASS: TestTerraformMLInfrastructure/ValidateVPC (2.15s)
    --- PASS: TestTerraformMLInfrastructure/ValidateEC2 (3.21s)
    --- PASS: TestTerraformMLInfrastructure/ValidateS3 (1.98s)
    --- PASS: TestTerraformMLInfrastructure/ValidateIAM (0.45s)
PASS
ok      terraform-ml-infrastructure-test    420.367s
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Terraform Tests

on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
        with:
          go-version: '1.19'
      - name: Run validation tests
        run: |
          cd tests
          go test -v -run TestTerraformValidation
      - name: Run integration tests
        if: github.event_name == 'push'
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          cd tests
          go test -v -timeout 45m
```

## Additional Resources

- [Terratest Documentation](https://terratest.gruntwork.io/)
- [Terraform Testing Guide](https://www.terraform.io/docs/language/modules/testing-experiment.html)
- [AWS Testing Best Practices](https://aws.amazon.com/blogs/devops/testing-infrastructure-as-code/)
