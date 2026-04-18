@echo off
cd /d C:\Users\Morne\Projects\PRISM
git init
git add arc.py README.md .env.example .gitignore LICENSE
git commit -m "Initial release: ARC - Adversarial Reasoning Chain"
git branch -M main
git remote add origin https://github.com/Morne-Ingstar/ARC.git
git push -u origin main
echo DONE
