@echo off
echo Đang chuẩn bị đẩy code lên GitHub...
git add .
git commit -m "Cập nhật chẩn đoán Viêm quanh khớp vai kèm mã ICD-10 chuẩn hóa và link tra cứu"
git push
echo.
echo Hoàn tất! Nhấn phím bất kỳ để thoát.
pause
