#!/bin/bash
# Ubuntu NUC系统优化脚本

# 检查root权限
if [ "$EUID" -ne 0 ]; then
  echo "请使用sudo运行此脚本"
  exit 1
fi

echo "开始优化NUC系统设置..."

# 设置CPU性能模式
echo "设置CPU性能模式..."
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1

# 禁用CPU节能模式
echo "禁用CPU节能模式..."
echo 0 | tee /proc/sys/kernel/nmi_watchdog > /dev/null 2>&1

# 优化内存管理
echo "优化内存管理..."
echo 3 | tee /proc/sys/vm/drop_caches > /dev/null 2>&1
echo 1 | tee /proc/sys/vm/compact_memory > /dev/null 2>&1

# 禁用透明大页
echo "禁用透明大页..."
echo never | tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null 2>&1
echo never | tee /sys/kernel/mm/transparent_hugepage/defrag > /dev/null 2>&1

# 禁用不必要的服务
echo "禁用不必要的服务..."
systemctl stop bluetooth.service > /dev/null 2>&1
systemctl stop cups.service > /dev/null 2>&1
systemctl stop avahi-daemon.service > /dev/null 2>&1
systemctl stop ModemManager.service > /dev/null 2>&1

# 提高实时优先级限制
echo "优化实时优先级限制..."
if ! grep -q "* - rtprio 99" /etc/security/limits.conf; then
  echo "* - rtprio 99" >> /etc/security/limits.conf
  echo "* - nice -20" >> /etc/security/limits.conf
fi

# 优化网络设置
echo "优化网络设置..."
echo 16777216 | tee /proc/sys/net/core/rmem_max > /dev/null 2>&1
echo 16777216 | tee /proc/sys/net/core/wmem_max > /dev/null 2>&1

# 优化磁盘设置
echo "优化磁盘设置..."
for disk in $(lsblk -d -o NAME | grep -v NAME); do
  blockdev --setra 4096 /dev/$disk > /dev/null 2>&1
done

# 优化NVME SSD（如果存在）
if [ -d "/sys/class/nvme" ]; then
  echo "优化NVME SSD设置..."
  for nvme in /dev/nvme*; do
    if [ -b "$nvme" ]; then
      echo 0 > /sys/block/$(basename $nvme)/queue/add_random
      echo 256 > /sys/block/$(basename $nvme)/queue/nr_requests
      echo 0 > /sys/block/$(basename $nvme)/queue/rotational
    fi
  done
fi

echo "系统优化完成！"