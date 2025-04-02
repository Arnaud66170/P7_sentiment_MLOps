@echo off
echo 🔄 Push vers Hugging Face...

cd huggingface_api
git add .
git commit -m "🚀 Update Hugging Face Space"
git push
cd ..

echo 🔄 Push vers GitHub...

git add .
git commit -m "📦 Sync GitHub repo"
git push origin main

echo ✅ Push complet terminé !
pause
