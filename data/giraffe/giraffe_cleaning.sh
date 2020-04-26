#!/bin/bash
sed -e '1,4d' negsamples.html > output.txt
sed 's/<img src="//' output.txt > output3.txt
sed 's/" .*$//' output3.txt > output4.txt
sed -i '$ d' output4.txt