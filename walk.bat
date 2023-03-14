@echo off
set KMP_DUPLICATE_LIB_OK=TRUE 
if [%1]==[] goto help

set args=%1 %2 %3 %4 %5 %6 %7 %8 %9
for %%q in (1 2 3 4 5 6 7 8 9 10) do shift 
set args=%args% %0 %1 %2 %3 %4 %5 %6 %7 %8 %9

python src/latwalk.py -v  %args%

ffmpeg -v warning -y -i _out\%~n1\%%05d.jpg _out\%~n1-%2%3%4%5%6%7%8%9.mp4
goto end

:help
echo Usage: walk -t textfile -im images [...]
:end
