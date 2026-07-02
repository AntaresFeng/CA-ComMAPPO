#!/bin/bash

# 获取脚本绝对路径
SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_FOLDER=$(dirname "$SCRIPT_PATH")
PROJECT_ROOT=$(dirname "$SCRIPT_FOLDER")

# 提取项目根目录文件夹名
ROOT_DIR_NAME=$(basename "$PROJECT_ROOT")

# 断言校验：根目录必须为 CA-ComMAPPO
if [[ "$ROOT_DIR_NAME" != "CA-ComMAPPO" ]]; then
    echo "❌ 校验失败：项目根目录名称应为 CA-ComMAPPO，当前是 $ROOT_DIR_NAME"
    echo "脚本路径：$SCRIPT_PATH"
    echo "识别到的项目根目录：$PROJECT_ROOT"
    exit 1
fi

# 进入项目根目录
cd "$PROJECT_ROOT" || {
    echo "错误：无法进入项目根目录 $PROJECT_ROOT"
    exit 1
}
echo "✅ 项目根目录校验通过：$PROJECT_ROOT"

echo -e "\n===== 递归清理全部 __pycache__ ====="
find . -type d -name "__pycache__" -exec rm -rf {} +

echo -e "\n===== 清理项目顶层 .*_cache 目录（仅根目录） ====="
find . -maxdepth 1 -type d -name ".*_cache" -exec rm -rf {} +

echo -e "\n✅ 缓存清理完成"