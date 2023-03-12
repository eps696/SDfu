@echo off
if [%1]==[] goto help
echo .. %1

python src/train.py --token "%1" --term "%2" --data data/%1 ^
--term_data data/%2 ^
%3 %4 %5 %6 %7 %8 %9

goto end

:help
echo Usage: train "<token>" category
echo  e.g.: train "<queen1>" lady
echo    or: train "<stripes1>" pattern --style
:end
