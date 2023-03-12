@echo off
set KMP_DUPLICATE_LIB_OK=TRUE 
if [%1]==[] goto help
echo .. %1

python src/gen.py -v -t %1 ^
%2 %3 %4 %5 %6 %7 %8 %9

goto end

:help
echo Usage: txt "text prompt" [...]
echo    or: txt textfile [...]
:end
