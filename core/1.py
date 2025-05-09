import random


def random_floating_value(number):
    fluctuation = random.uniform(0.95, 1.05)
    return number * fluctuation


print("数字浮动工具（输入 exit 或直接回车退出）")

while True:
    user_input = input("\n请输入数字: ").strip()

    # 退出条件
    if user_input.lower() in ('exit', 'quit', ''):
        print("程序已退出")
        break

    try:
        number = float(user_input)
        result = random_floating_value(number)
        print(f"原始值: {number:.2f} → 浮动后: {result:.2f} （波动范围 ±5%）")
    except ValueError:
        print("错误：请输入有效数字！")
        continue

# 可选功能：最终统计（添加在循环外）
# print(f"\n共处理了 {counter} 次有效输入")