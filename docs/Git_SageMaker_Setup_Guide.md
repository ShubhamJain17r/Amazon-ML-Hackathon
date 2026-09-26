# Git Setup Guide & Environment Notice
### Repository: `git@github.com:ShubhamJain17r/Amazon-ML-Hackathon.git`

> [!IMPORTANT]
> **COMPUTE ARCHITECTURE UPDATE:**  
> Due to AWS default quota limitations on SageMaker notebook instances (`ResourceLimitExceeded: ml.m5.xlarge quota is 0`), our team is **exclusively using AWS S3 for central cloud data storage**. Heavy compute is run on **Google Colab or Local machines**.  
> 👉 **For the latest setup instructions, see:** [`Team_Setup_Colab_Git_S3_Guide.md`](file:///home/shubham/Projects/Amazon%20ML%20Hackathon/docs/Team_Setup_Colab_Git_S3_Guide.md).

---

## Setting Up Git (Terminal / Local Machine / Colab)

Every team member must configure their Git identity:

### Step 1 — Configure Your Git Identity
```bash
git config --global user.name "Your Full Name"
git config --global user.email "your_github_email@example.com"
```

Verify your configuration:
```bash
git config --global --list
```

---

### Step 2 — Generate an SSH Key (For Local Linux / Mac Machines)

Run this command in the terminal (replace with your GitHub email):
```bash
ssh-keygen -t ed25519 -C "your_github_email@example.com"
```

- When prompted: `Enter file in which to save the key:` Press Enter (accept default).
- When prompted: `Enter passphrase:` Press Enter twice (leave empty for passwordless push/pull).

Now display your public SSH key:
```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the output line starting with `ssh-ed25519 ...` and add it to your GitHub profile at **[github.com/settings/keys](https://github.com/settings/keys)**.

Verify connection:
```bash
ssh -T git@github.com
```

---

### Step 3 — Daily Team Workflow (Avoiding Conflicts)

1. **Before beginning work each day, always pull latest updates:**
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

> [!CAUTION]
> Large files like `.tsv` and `.parquet` are automatically ignored by `.gitignore`. **Always store and retrieve data from our central AWS S3 bucket!**
