#!/bin/sh
ssh_connection() {
    SSH_FILE="$HOME/.ssh/github_deploy_key_kuber"
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Decrypt the file containing the private key
    openssl aes-256-cbc \
        -K $encrypted_73348c0d1a75_key \
        -iv $encrypted_73348c0d1a75_iv \
        -in "github_deploy_key_kuber.enc" \
        -out "$SSH_FILE" -d
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Enable SSH authentication
    chmod 600 "$SSH_FILE" 
    ssh-add "$SSH_FILE" 
}

setup_git() {
  git config --global user.email "$GH_USER_EMAIL"
  git config --global user.name "$GH_USER_NAME"
}

commit_files() {
  git checkout --quiet -b add-badges
  git add .
  git commit --message "Update coverage badge - build: $TRAVIS_BUILD_NUMBER"
}

upload_files(){
  git push --quiet --force git@github.com:"${TRAVIS_REPO_SLUG}".git add-badges
}

ssh_connection
setup_git
commit_files
upload_files