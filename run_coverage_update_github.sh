#!/bin/bash

set -e

# Run tests, generate reports
# coverage run -m pytest tests
# coverage report -m
# coverage-badge -f -o coverage.svg
#pylint loa || pylint-exit $?

# Configure Git & SSH
git config --global user.email "$GH_USER_EMAIL"
git config --global user.name "$GH_USER_NAME"

eval "$(ssh-agent)"

SSH_FILE="$HOME/.ssh/github_deploy_key_pyloa"
mkdir -p $(dirname $SSH_FILE)
ssh-keyscan -t rsa github.com > $(dirname $SSH_FILE)/known_hosts

openssl aes-256-cbc -K $encrypted_73348c0d1a75_key -iv $encrypted_73348c0d1a75_iv -in "github_deploy_key_kuber.enc" -out "$SSH_FILE" -d
chmod 600 "$SSH_FILE" 
ssh-add "$SSH_FILE" 

# Push changes to Github
git checkout --quiet -b add-badges
git add .
git commit --message "Update coverage badge - build: $TRAVIS_BUILD_NUMBER"

git push --quiet --force git@github.com:"${TRAVIS_REPO_SLUG}".git add-badges