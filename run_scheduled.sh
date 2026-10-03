#!/bin/bash
set -euo pipefail
cd /Users/jasonlei/Autovisor-account-b

# 原账号尚未退出时等待，避免两个账号的自动任务同时运行。
while /bin/launchctl print "gui/$(id -u)/com.autovisor.daily" 2>/dev/null | /usr/bin/grep -q 'state = running'; do
    echo "[$(/bin/date '+%Y-%m-%d %H:%M:%S')] 原账号仍在运行，新账号等待 60 秒。"
    /bin/sleep 60
done

exec /bin/bash ./run_macos.sh --config /Users/jasonlei/Autovisor-account-b/config.macos.ini
