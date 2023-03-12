@echo off
set KMP_DUPLICATE_LIB_OK=TRUE 
if [%1]==[] goto help
echo .. %1 .. %2

if exist _in\%~n1\* goto proc
set seq=ok
echo .. making source sequence
mkdir _in\%~n1
ffmpeg -y -v warning -i _in\%1 -q:v 2 _in\%~n1\%%06d.jpg

:proc
echo .. processing
python src/gen.py -v -im _in/%~n1 -o _out/%~n1 -t %2 ^
%3 %4 %5 %6 %7 %8 %9

if %seq%==ok goto seq
goto end

:seq
ffmpeg -y -v warning -i _out\%~n1\%%06d.jpg _out\%~n1-%2-%3%4%5%6%7%8%9.mp4
goto end

:help
echo Usage: img imagedir "text prompt" [...]
echo    or: img videofile "text prompt" [...]
:end
