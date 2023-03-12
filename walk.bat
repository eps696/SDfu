@echo off
set KMP_DUPLICATE_LIB_OK=TRUE 
if [%1]==[] goto help
echo .. %1

python src/latwalk.py -v -t %1 ^
%2 %3 %4 %5 %6 %7 %8 %9

ffmpeg -v warning -y -i _out\%~n1\%%06d.jpg _out\%~n1-%2%3%4%5%6%7%8%9.mp4
goto end

:help
echo Usage: walk textfile [...]
:end
