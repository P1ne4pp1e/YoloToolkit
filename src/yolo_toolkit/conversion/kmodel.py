#!/usr/bin/env python3
"""
K210直接转换脚本 - 适用于已量化的ONNX模型
不需要校准数据集，直接转换
"""

import os
import subprocess
import sys


def check_nncase_version():
    """检查nncase版本"""
    try:
        import nncase
        version = getattr(nncase, '__version__', '未知')
        print(f"当前nncase版本: {version}")
        return True
    except ImportError:
        print("❌ nncase未安装")
        return False


def convert_with_python_api():
    """使用Python API直接转换（无量化）"""
    try:
        import nncase
        print("🔄 使用Python API转换...")

        # 最简单的编译选项
        compile_options = nncase.CompileOptions()
        compile_options.target = "k210"
        compile_options.dump_ir = False
        compile_options.dump_asm = False

        # 创建编译器
        compiler = nncase.Compiler(compile_options)

        # 读取已量化的ONNX模型
        with open('model_quant.onnx', 'rb') as f:
            model_content = f.read()

        # 导入选项
        import_options = nncase.ImportOptions()

        # 导入模型
        compiler.import_onnx(model_content, import_options)

        # 直接编译（不进行量化）
        compiler.compile()

        # 生成kmodel
        kmodel = compiler.gencode_tobytes()

        # 保存
        with open('output.kmodel', 'wb') as f:
            f.write(kmodel)

        print("✅ Python API转换成功！")
        return True

    except Exception as e:
        print(f"❌ Python API转换失败: {e}")
        return False


def convert_with_ncc_no_dataset():
    """使用ncc命令行转换（无数据集）"""
    print("🔄 使用ncc命令行转换...")

    # 不使用dataset参数的ncc命令
    cmd = "ncc compile -i onnx -t k210 model_quant.onnx output.kmodel"

    print(f"执行命令: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ ncc命令转换成功！")
            if result.stdout:
                print("输出:", result.stdout)
            return True
        else:
            print(f"❌ ncc命令失败")
            if result.stderr:
                print("错误信息:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ ncc命令执行失败: {e}")
        return False


def convert_with_legacy_api():
    """使用旧版API语法转换"""
    try:
        import nncase
        print("🔄 使用旧版API语法转换...")

        # 方法1：直接指定目标字符串
        try:
            compiler = nncase.Compiler("k210")

            # 读取模型
            with open('model_quant.onnx', 'rb') as f:
                model_content = f.read()

            # 导入模型
            compiler.import_onnx(model_content)

            # 编译
            compiler.compile()

            # 保存
            with open('output.kmodel', 'wb') as f:
                f.write(compiler.gencode_tobytes())

            print("✅ 旧版API转换成功！")
            return True

        except Exception as e1:
            print(f"旧版API方法1失败: {e1}")

            # 方法2：尝试其他旧版语法
            try:
                compile_options = nncase.CompileOptions()
                compile_options.target = "k210"

                # 禁用量化相关选项
                compile_options.preprocess = False

                compiler = nncase.Compiler(compile_options)

                with open('model_quant.onnx', 'rb') as f:
                    model_content = f.read()

                compiler.import_onnx(model_content)
                compiler.compile()

                with open('output.kmodel', 'wb') as f:
                    f.write(compiler.gencode_tobytes())

                print("✅ 旧版API方法2转换成功！")
                return True

            except Exception as e2:
                print(f"旧版API方法2失败: {e2}")
                return False

    except Exception as e:
        print(f"❌ 旧版API转换失败: {e}")
        return False


def provide_installation_guide():
    """提供安装指南"""
    print("\n🛠️  nncase版本兼容性解决方案:")
    print("=" * 50)

    print("方案1: 卸载重装兼容版本")
    print("pip uninstall nncase -y")
    print("pip install nncase==1.4.0  # 或其他支持K210的版本")

    print("\n方案2: 手动下载特定版本")
    print("1. 访问: https://github.com/kendryte/nncase/releases")
    print("2. 下载支持K210的版本 (通常是v0.2.x - v1.x)")
    print("3. pip install <下载的wheel文件>")

    print("\n方案3: 使用现有版本的命令行工具")
    print("ncc compile -i onnx -t k210 model_quant.onnx output.kmodel")

    print("\n方案4: 检查是否有ncc可执行文件")
    print("which ncc  # Linux/Mac")
    print("where ncc  # Windows")


def main():
    print("🚀 K210直接转换工具（已量化模型）")
    print("=" * 50)

    # 检查模型文件
    if not os.path.exists('model_quant.onnx'):
        print("❌ 未找到 model_quant.onnx 文件")
        print("请确保已量化的ONNX模型文件在当前目录")
        return

    # 显示模型信息
    model_size = os.path.getsize('model_quant.onnx') / (1024 * 1024)
    print(f"📄 模型文件: model_quant.onnx ({model_size:.2f} MB)")

    # 检查nncase是否安装
    if not check_nncase_version():
        print("请先安装nncase")
        provide_installation_guide()
        return

    print("\n🎯 开始转换（跳过量化步骤）...")

    # 尝试多种转换方法
    methods = [
        ("ncc命令行（无数据集）", convert_with_ncc_no_dataset),
        ("Python API（简化版）", convert_with_python_api),
        ("旧版API语法", convert_with_legacy_api),
    ]

    success = False
    for method_name, method_func in methods:
        print(f"\n🔄 尝试方法: {method_name}")
        try:
            if method_func():
                success = True
                break
        except Exception as e:
            print(f"❌ {method_name} 出错: {e}")

    if success:
        print(f"\n🎉 转换成功！")

        if os.path.exists('output.kmodel'):
            kmodel_size = os.path.getsize('output.kmodel') / (1024 * 1024)
            print(f"✅ 生成文件: output.kmodel ({kmodel_size:.2f} MB)")

            # 大小检查
            if kmodel_size > 6:
                print("⚠️  警告: 模型超过6MB，可能超出K210硬件限制")
            elif kmodel_size > 2:
                print("⚠️  提示: 模型大于2MB，在MaixPy环境可能内存不足")
            else:
                print("✅ 模型大小适合K210部署")

        print(f"\n📋 部署步骤:")
        print("1. 将output.kmodel拷贝到K210设备")
        print("2. 使用KPU API加载: kpu.load('output.kmodel')")
        print("3. 设置输入数据并推理")

    else:
        print(f"\n💡 所有转换方法都失败了")
        print("这通常是nncase版本兼容性问题")
        provide_installation_guide()

        print(f"\n🔧 手动命令尝试:")
        print("ncc compile -i onnx -t k210 model_quant.onnx output.kmodel")


if __name__ == "__main__":
    main()