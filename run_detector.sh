#!/bin/bash
# 目标检测启动脚本

# 默认配置
MODEL="yolo11n.onnx"
DEVICE="AUTO"
WIDTH=960
HEIGHT=540
CONF=0.4
OPTIMIZE=false
DISPLAY=true

export QT_QPA_PLATFORM=xcb
# 在 QT_QPA_PLATFORM=xcb 行之后添加
export DISPLAY=:0

# 使用说明
function show_usage {
  echo "使用方法: $0 [选项]"
  echo "选项:"
  echo "  -m, --model <文件>     模型文件路径 (默认: yolo11n.onnx)"
  echo "  -d, --device <设备>    推理设备: CPU, GPU, MYRIAD, AUTO (默认: AUTO)"
  echo "  -w, --width <像素>     图像宽度 (默认: 960)"
  echo "  -h, --height <像素>    图像高度 (默认: 540)"
  echo "  -c, --conf <阈值>      置信度阈值 0.0-1.0 (默认: 0.4)"
  echo "  -o, --optimize         应用系统优化 (需要root权限)"
  echo "  -n, --no-display       无显示模式"
  echo "  --help                 显示此帮助信息"
  exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL="$2"
      shift 2
      ;;
    -d|--device)
      DEVICE="$2"
      shift 2
      ;;
    -w|--width)
      WIDTH="$2"
      shift 2
      ;;
    -h|--height)
      HEIGHT="$2"
      shift 2
      ;;
    -c|--conf)
      CONF="$2"
      shift 2
      ;;
    -o|--optimize)
      OPTIMIZE=true
      shift
      ;;
    -n|--no-display)
      DISPLAY=false
      shift
      ;;
    --help)
      show_usage
      ;;
    *)
      echo "未知选项: $1"
      show_usage
      ;;
  esac
done

# 检查模型文件
if [ ! -f "$MODEL" ]; then
  echo "错误: 模型文件不存在: $MODEL"
  exit 1
fi

# 显示配置信息
echo "=== YOLO目标检测 ==="
echo "模型文件: $MODEL"
echo "推理设备: $DEVICE"
echo "分辨率: ${WIDTH}x${HEIGHT}"
echo "置信度阈值: $CONF"
echo "系统优化: $OPTIMIZE"
echo "显示模式: $DISPLAY"

# 应用系统优化
if [ "$OPTIMIZE" = true ]; then
  echo "正在应用系统优化..."
  if [ -f "./system_optimize.sh" ]; then
    sudo ./system_optimize.sh
  else
    echo "警告: 系统优化脚本不存在"
  fi
fi

# 构建命令
CMD="python openvino_async.py --model $MODEL --device $DEVICE --width $WIDTH --height $HEIGHT --conf $CONF"

# 添加额外参数
if [ "$DISPLAY" = false ]; then
  CMD="$CMD --no-display"
fi

if [ "$OPTIMIZE" = true ]; then
  CMD="$CMD --optimize"
fi

# 设置CPU亲和性
if command -v taskset &> /dev/null; then
  # 使用所有CPU核心
  # 在run_detector.sh中
  CMD="taskset -c 0-15 $CMD"  # 使用所有16个核心
  echo "已启用CPU亲和性"
fi

# 运行命令
echo "执行命令: $CMD"
eval $CMD

exit $?