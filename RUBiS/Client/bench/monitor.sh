#!/bin/sh

/usr/bin/ssh $1 /usr/bin/sar -n DEV -n SOCK -rubwdFB $2 $3 > $4
