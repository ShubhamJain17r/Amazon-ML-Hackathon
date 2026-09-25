# Git & GitHub Setup Guide inside SageMaker JupyterLab
### Repository: `git@github.com:ShubhamJain17r/Amazon-ML-Hackathon.git`

Every team member should run these steps once inside their own SageMaker JupyterLab instance.

---

## Step 1 — Open the Terminal in JupyterLab

1. Open your SageMaker JupyterLab interface.
2. In the top menu, click: **File → New → Terminal**
3. Verify Git is installed:
```bash
git --version
```
*(You should see `git version 2.x.x`)*

---

## Step 2 — Configure Your Git Identity

Set your personal name and the email associated with your GitHub account (each teammate should put their own name and email):

```bash
git config --global user.name "Your Full Name"
git config --global user.email "your_github_email@example.com"
```

Verify your configuration:
```bash
git config --global --list
```

---

## Step 3 — Generate an SSH Key inside SageMaker

Run this command in the terminal (replace with your GitHub email):

```bash
ssh-keygen -t ed25519 -C "your_github_email@example.com"
```

- When prompted: `Enter file in which to save the key (/home/ec2-user/.ssh/id_ed25519):`  
  👉 **Press Enter** (accept default).
- When prompted: `Enter passphrase (empty for no passphrase):`  
  👉 **Press Enter twice** (leave empty for seamless git push/pull without password prompts).

Now display your public SSH key:
```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the entire output line that starts with `ssh-ed25519 ...`.

---

## Step 4 — Add the SSH Key to Your GitHub Account

1. Open GitHub in your web browser: **[github.com/settings/keys](https://github.com/settings/keys)**
2. Click the green button: **New SSH key**
3. **Title:** `AWS SageMaker - Amazon ML Challenge`
4. **Key type:** `Authentication Key`
5. **Key:** Paste the copied public key.
6. Click **Add SSH key**.

Verify the connection from your SageMaker terminal:
```bash
ssh -T git@github.com
```
*(Type `yes` if asked about host authenticity).*  
You should see:
> `Hi <YourUsername>! You've successfully authenticated, but GitHub does not provide shell access.`

---

## Step 5 — Clone the Team Repository

> **Important:** Always navigate to `~/SageMaker` before cloning so your files **persist** across notebook instance stops and restarts!

```bash
cd ~/SageMaker
git clone git@github.com:ShubhamJain17r/Amazon-ML-Hackathon.git
```

Move into the repository folder:
```bash
cd Amazon-ML-Hackathon
```

Verify repository status:
```bash
git status
```
Output should be:
```text
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

---

## Step 6 — Daily Team Workflow (Avoiding Conflicts)

1. **Before you begin work each day, always pull the latest updates:**
```bash
git pull origin main
```

2. **Work strictly inside your designated personal folder:**
- Shubham: `notebooks/Shubham/`
- Karan: `notebooks/Karan/`
- Suhani: `notebooks/Suhani/`
- Vishal: `notebooks/Vishal/`

3. **To commit and push your work:**
```bash
git add notebooks/<YourName>/
git commit -m "feat: added eda and candidate generation tests"
git push origin main
```

*(Large files like `.tsv` and `.parquet` are automatically ignored by `.gitignore` — always keep data stored in S3!)*
