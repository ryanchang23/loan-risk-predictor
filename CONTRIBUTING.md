# 📘 How to Contribute to This Project

Thank you for your interest in contributing to this project! This guide will walk you through the full process—from installing Git to submitting a Pull Request (PR).

---

## 1. Install Git

If you don’t already have Git installed, follow the steps below:

### On Windows:

1. Download the Git installer from: [https://git-scm.com/download/win](https://git-scm.com/download/win)  
2. Run the installer and proceed with the default settings.

### On macOS or Linux:

```bash
# macOS
brew install git

# Ubuntu / Debian
sudo apt update && sudo apt install git

# Arch / Manjaro
sudo pacman -S git

```

## 2. Configure Git User Info
```bash
git config --global user.name "your name"
git config --global user.email "your email"
```

## 3. Fork the Project
Go to the GitHub repository page and click on the Fork button in the top-right corner to create a copy under your account.

## 4. Clone Your Fork 
``` bash
# https
git clone https://github.com/<your_account>/loan-risk-predictor.git
cd project_name

# SSH (recommended)
git clone git@github.com:<your_account>/loan-risk-predictor.git
cd project_name

```

## 5. Add the Original Repository as a Remote (upstream)
This allows you to keep your fork up to date with the main project:

``` bash

git remote add upstream https://github.com/JiaLong0209/loan-risk-predictor.git

# Verify remotes
git remote -v
``` 

## 6. Create a Feature Branch
Do not work directly on the main or master branch. Instead, create a new feature branch:

``` bash
git checkout -b feat/your-feature-name
``` 

## 7. Commit and Push Changes
After editing the code:

``` bash
git add .

# Commit your changes
git commit -m "feat: add your feature description"

# Push to your GitHub  repository
git push origin feat/your-feature-name
```

## 8. Submit ta Pull Request (PR)

1. Go to your GitHub repo.

2. Click the "Compare & pull request" button.

3. Add a meaningful description and submit the PR.

4. Wait for a maintainer to review and merge it.


## 9. Keep Your Fork in Sync
Regularly sync your fork with the main repository to stay updated:

```bash
git checkout main
git pull upstream main
git push origin main
```

## 10. Delete Merged Branches (Optional)

Once your PR is merged, you can clean up your branches:

``` bash
# Delete the local branch
git branch -d feat/your-feature-name

# Delete the remote branch
git push origin --delete feat/your-feature-name

```

## Additional

### Using SSH Key for Git Operations (Recommended)

#### Generate SSH Key:

``` bash    
ssh-keygen -t rsa -C "your email"

cat <path_to_id_rsa.pub>

```

Copy the output.

#### Add SSH Key to GitHub:

1. Log in to GitHub.

2. Go to Settings -> SSH and GPG Keys.

3. Click Neww SSH Key.

4. Paste the copied key and save.

#### Verify the SSH:
``` bash    
ssh -T git@github.com
```

### Change Git remote URL from HTTPS to SSH

#### Check your current remotes

```bash
git remote -v
```

#### Change the remote URL to SSH

``` bash
git remote set-url origin git@github.com:your-username/your-repo.git
```

#### Verify the change

```bash
git remote -v
```

#### Test your connection

```bash
ssh -T git@github.com   
```


