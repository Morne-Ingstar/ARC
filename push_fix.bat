@echo off
cd /d C:\Users\Morne\Projects\PRISM
copy /Y arc.py prism.py
git add arc.py
git commit -m "fix: enum-based Gemini error handling per ARC audit"
git push origin main
