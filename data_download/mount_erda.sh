#!/bin/bash
key=/home/sxr280/.ssh/id_rsa
user=junwang@bio.ku.dk
erdadir=/
mnt=/tmp/erda
if [ -f "$key" ]
then
    mkdir -p ${mnt}
    sshfs ${user}@io.erda.dk:${erdadir} ${mnt} -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 -o IdentityFile=${key} 
else
    echo "'${key}' is not an ssh key"
fi
# oswin.krause@di.ku.dk@io.erda.dk:/ /tmp/zbh251-erda -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,IdentityFile=/home/zbh251/.ssh/id_rsa
