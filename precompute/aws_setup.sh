#!/bin/bash
# One-time AWS setup for blueprint precomputation.
#
# Creates: S3 bucket, security group, IAM role, SSH key pair.
# Run this ONCE before launch_blueprint.sh.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Sufficient IAM permissions to create resources

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-poker-solver-blueprints}"
KEY_NAME="${KEY_NAME:-poker-solver-key}"
SECURITY_GROUP="${SECURITY_GROUP:-poker-solver-sg}"
ROLE_NAME="poker-solver-instance-role"
PROFILE_NAME="poker-solver-instance-profile"

echo "=== AWS Setup for Blueprint Precompute ==="
echo "Region: $REGION"
echo ""

# ── S3 Bucket ────────────────────────────────────────────────────────────

echo "1. Creating S3 bucket: $S3_BUCKET"
if aws s3 ls "s3://$S3_BUCKET" 2>/dev/null; then
    echo "   Already exists."
else
    if [ "$REGION" = "us-east-1" ]; then
        aws s3 mb "s3://$S3_BUCKET" --region "$REGION"
    else
        aws s3 mb "s3://$S3_BUCKET" --region "$REGION" \
            --create-bucket-configuration LocationConstraint="$REGION"
    fi
    echo "   Created."
fi

# ── SSH Key Pair ─────────────────────────────────────────────────────────

echo "2. Creating SSH key pair: $KEY_NAME"
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" 2>/dev/null; then
    echo "   Already exists."
else
    aws ec2 create-key-pair --key-name "$KEY_NAME" --region "$REGION" \
        --query "KeyMaterial" --output text > "${KEY_NAME}.pem"
    chmod 400 "${KEY_NAME}.pem"
    echo "   Created. Private key saved to ${KEY_NAME}.pem"
fi

# ── Security Group ───────────────────────────────────────────────────────

echo "3. Creating security group: $SECURITY_GROUP"
SG_ID=$(aws ec2 describe-security-groups \
    --group-names "$SECURITY_GROUP" --region "$REGION" \
    --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "")

if [ -n "$SG_ID" ] && [ "$SG_ID" != "None" ]; then
    echo "   Already exists: $SG_ID"
else
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP" \
        --description "Poker solver blueprint precompute" \
        --region "$REGION" \
        --query "GroupId" --output text)
    # Allow SSH from anywhere (for debugging; tighten in production)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" --region "$REGION" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0
    echo "   Created: $SG_ID"
fi

# ── IAM Role (for S3 access from instances) ──────────────────────────────

echo "4. Creating IAM role: $ROLE_NAME"
TRUST_POLICY='{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "ec2.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}'

if aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null; then
    echo "   Role already exists."
else
    aws iam create-role --role-name "$ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" > /dev/null
    aws iam attach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
    echo "   Created with S3 access."
fi

# Instance profile
if aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" 2>/dev/null; then
    echo "   Instance profile already exists."
else
    aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" > /dev/null
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
    echo "   Instance profile created."
    echo "   Waiting 10s for IAM propagation..."
    sleep 10
fi

echo ""
echo "=== Setup Complete ==="
echo "S3 bucket:  $S3_BUCKET"
echo "Key pair:   $KEY_NAME"
echo "Sec group:  $SECURITY_GROUP ($SG_ID)"
echo "IAM role:   $ROLE_NAME"
echo ""
echo "Next: ./launch_blueprint.sh"
