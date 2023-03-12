@echo off
set KMP_DUPLICATE_LIB_OK=TRUE 
if [%1]==[] goto help
echo .. %1 .. %2 .. %3

python src/gen.py -v -im %1 -o _out/%~n1 --mask %2 -t %3 ^
%4 %5 %6 %7 %8 %9

goto end

:help
echo Usage: inpaint imagedir masksdir "text prompt" [...]
echo  e.g.: inpaint _in/pix _in/pix/mask "steampunk fantasy" 
echo    or: inpaint _in/pix "human figure" "steampunk fantasy" -m 15i
:end
