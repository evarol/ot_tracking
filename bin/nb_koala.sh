#!/bin/bash
#
# Script for running Jupyter Lab on Koala

WORKDIR='/home/mn2822/Desktop/WormOT/ot_tracking'
CONDA_ENV='worm-ot'
LOCAL_PORT="8888"
REMOTE_PORT="9000"
UNAME="mn2822"
HOSTNAME="koala.paninski.zi.columbia.edu"

cmd_start() {

    cmd_1="cd ${WORKDIR}"
    cmd_2="source ~/anaconda3/etc/profile.d/conda.sh"
    cmd_3="conda activate ${CONDA_ENV}"
    cmd_4="jupyter lab --no-browser --port ${REMOTE_PORT} > /dev/null &"
    cmds="${cmd_1}; ${cmd_2}; ${cmd_3}; ${cmd_4}"

    ssh ${UNAME}@${HOSTNAME} -T ${cmds}
}

cmd_stop() {

    ssh ${UNAME}@${HOSTNAME} -T "pkill -u ${UNAME} jupyter"
}

cmd_connect() {

    ssh -N -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} ${UNAME}@${HOSTNAME}
}

case $1 in
    
    "start")
        cmd_start
        ;;

    "stop")
        cmd_stop
        ;;

	"connect")
		cmd_connect
		;;

    *)
        echo "Usage: koala_nb <subcommand>"
        echo "Subcommands:"
        echo "    start: Start remote notebook"
        echo "    stop: Stop remote notebook"
        echo "    connect: Connect to remote notebook with port-forwarding"
        ;;

esac
