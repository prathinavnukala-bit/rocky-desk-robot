#!/bin/bash
# Rocky brain control:  ./rocky_ctl.sh start | stop | restart | status | logs | diagnose
# Keeps the Mac awake (caffeinate) for as long as the brain runs, so Rocky never
# goes deaf because the Mac fell asleep.
DIR="$(cd "$(dirname "$0")" && pwd)"
PY=/usr/local/bin/python3
LOG=/tmp/rocky_brain.log
PIDF=/tmp/rocky_brain.pid

running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

case "$1" in
  start)
    running && { echo "Rocky brain already running (pid $(cat $PIDF))"; exit 0; }
    cd "$DIR" && nohup caffeinate -ims "$PY" -u brain.py >> "$LOG" 2>&1 &
    echo $! > "$PIDF"; echo "Started (pid $!). Log: $LOG" ;;
  stop)
    running && kill "$(cat $PIDF)" 2>/dev/null
    pkill -f "[Pp]ython.* -u brain\.py" 2>/dev/null   # the interpreter shows up as "Python -u brain.py"
    sleep 1
    pgrep -f "[Pp]ython.* -u brain\.py" >/dev/null && { pkill -9 -f "[Pp]ython.* -u brain\.py"; }
    pkill -f "stt_server\.py" 2>/dev/null
    rm -f "$PIDF"; echo "Stopped." ;;
  restart) "$0" stop; sleep 2; "$0" start ;;
  status) running && echo "running (pid $(cat $PIDF))" || echo "not running"; tail -3 "$LOG" 2>/dev/null ;;
  logs) tail -f "$LOG" ;;
  diagnose) cd "$DIR" && "$PY" diagnose.py "${@:2}" ;;
  *) echo "usage: $0 start|stop|restart|status|logs|diagnose [--deep|--soak N]"; exit 1 ;;
esac
