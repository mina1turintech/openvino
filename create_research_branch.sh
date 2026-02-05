#!/bin/bash
# Script to create and push the research branch for Qwen2-1.5B layout optimization
# This script should be run from the root of the OpenVINO repository

set -e  # Exit on error

BRANCH_NAME="research/layout-optimization-qwen2-1.5b"
BASE_BRANCH="develop"

echo "=================================="
echo "Creating Research Branch"
echo "=================================="
echo "Branch name: ${BRANCH_NAME}"
echo "Base branch: ${BASE_BRANCH}"
echo ""

# Step 1: Check current git status
echo "Step 1: Checking current git status..."
git status

# Step 2: Fetch latest changes from remote
echo ""
echo "Step 2: Fetching latest changes from remote..."
git fetch origin

# Step 3: Ensure we're on the develop branch and it's up to date
echo ""
echo "Step 3: Checking out ${BASE_BRANCH} branch..."
git checkout ${BASE_BRANCH}

echo "Step 4: Pulling latest changes for ${BASE_BRANCH}..."
git pull origin ${BASE_BRANCH}

# Step 4: Create new research branch
echo ""
echo "Step 5: Creating new branch ${BRANCH_NAME}..."
git checkout -b ${BRANCH_NAME}

# Step 5: Push the new branch to remote
echo ""
echo "Step 6: Pushing ${BRANCH_NAME} to remote..."
git push -u origin ${BRANCH_NAME}

# Step 6: Verify the branch was created
echo ""
echo "Step 7: Verifying branch creation..."
git branch -vv | grep ${BRANCH_NAME}

echo ""
echo "=================================="
echo "✓ Success!"
echo "=================================="
echo "Research branch '${BRANCH_NAME}' has been created and pushed to remote."
echo "The branch is now accessible to team members."
echo ""
echo "Current branch:"
git branch --show-current
echo ""
echo "To start working on this branch from another machine, use:"
echo "  git fetch origin"
echo "  git checkout ${BRANCH_NAME}"
