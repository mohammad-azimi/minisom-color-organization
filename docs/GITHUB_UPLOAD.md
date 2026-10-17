# Publish from your own computer

Open CMD inside the extracted project folder. Git and GitHub CLI (`gh`) must be
installed. Confirm the active account before creating the repository:

```bat
gh auth status --hostname github.com
```

If no account is connected, sign in locally:

```bat
gh auth login --hostname github.com
```

Create the local commit if the downloaded folder does not already contain Git
history:

```bat
git init -b main
git add .
git commit -m "Add reproducible MiniSom color organization project"
```

Create and push a public repository under the active GitHub account:

```bat
gh repo create minisom-color-organization --public --source=. --remote=origin --push
```

Use `--private` in place of `--public` if you want a private repository. If the
repository name already exists, choose a different name rather than overwriting
another project. Open the created repository with:

```bat
gh repo view --web
```

For later changes:

```bat
git add .
git commit -m "Update MiniSom experiment"
git push
```
