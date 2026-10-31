# Flow Transformer as a PDE foundation model
## 1. Set up SSH Key
Using Claude and [Gitlab docs](https://docs.gitlab.com/user/ssh/)
- Generate an SSH Key Pair (Replace the "your.email@example.com" with your email id)
```
ssh-keygen -t ed25519 -C "your.email@example.com"
```
- Copy your public key
```
type %USERPROFILE%\.ssh\id_ed25519.pub
```
- Add Your SSH Key to GitLab: Preferences -> SSH Keys -> Add Key -> Paste the copied key in "Key tab"
- Test your connection
```
ssh -T -p 10022 git@lisdi-git.lanl.gov
```
(After typing 'yes'. If it prints "Welcome to Gitlab "your-username", you are connected)

## 2. Clone the repository
To clone the repository, click on top-right 'code' and select 'clone with SSH' and copy the code path and paste in anaconda prompt.
or, open anaconda prompt and paste
```
git clone ssh://git@lisdi-git.lanl.gov:10022/mrautela/pde-flowformer.git
```
Go to the directory
```
cd pde-flowformer
```

## 3. Install the requirements
- Install conda
```
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
```
bash Miniconda3-latest-Linux-x86_64.sh
```
- Install dependencies via environment.yml
```
conda env create -f environment.yml
```
- Activate the environment
```
conda activate pytorch_py312_env
```
- Install pytorch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128                    
```
- Check pytorch installation
```
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```
Output: 
PyTorch version: <version>
CUDA available: True

## 4. Run the scripts

## 5. Download datasets

Install Hugging Face download tools:

Create directories for the PDEGym datasets:

```bash
mkdir -p datasets/pdegym/CE-RM
```

### Download CE-RM
```
hf download camlab-ethz/CE-RM \
  --repo-type dataset \
  --local-dir datasets/pdegym/CE-RM \
  --include "*.nc" \
  --include "assemble_data.py"
```


### Assemble CE-RM
```
python datasets/pdegym/CE-RM/assemble_data.py \
  --input_dir datasets/pdegym/CE-RM \
  --output_file datasets/pdegym/CE-RM/CE-RM.nc
```

After assembly, the expected files are:

```text
datasets/pdegym/CE-Gauss/CE-Gauss.nc
datasets/pdegym/CE-RM/CE-RM.nc
```

## 5. Demo run the script

---

## 6. Git workflow for making changes

Before making code changes, check the repository status:

```bash
git status
```

Update your local repository:

```bash
git pull
```

Create a new branch for your changes:

```bash
git checkout -b your-branch-name
```

After editing files, check which files changed:

```bash
git status
```

Add a specific file:

```bash
git add <file-name>
```

Or add all changed files:

```bash
git add .
```

Commit your changes with a clear message:

```bash
git commit -m "Describe your changes"
```

Push your new branch to GitLab:

```bash
git push -u origin your-branch-name
```

After pushing, open a merge request from your branch into the main branch.