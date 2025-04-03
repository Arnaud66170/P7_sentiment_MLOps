@echo off
echo 🔄 Push vers Hugging Face...

cd huggingface_api
git add .
git commit -m "🚀 Update Hugging Face Space"
git push origin main
cd ..

echo 🔄 Push vers GitHub...

git add .
git commit -m "📦 Sync GitHub repo"
git push github main

echo ✅ Push complet terminé !
pause
