# tensor_generator 使用指南

本文档面向第一次接触 tensor_generator 的使用者，通过具体示例介绍如何生成包含特殊值（Inf、NaN、DTypeMin、DTypeMax 等）的测试张量。

---

## 基本概念

tensor_generator 的核心思路是：**一个张量 = 背景分布 + 离散特殊值叠加**。

```
生成的张量 = 背景（连续/整数分布采样） + 按概率随机替换为指定的离散值
```

对应到代码中的两个参数：

- **`TensorSpec`** — 描述"哪些离散值、各占多大比例"
- **`BaseDist`** — 描述"背景用什么分布填充"

```cpp
#include "tensor_generator.h"

auto tensor = generate_tensor(
    shape,      // 张量形状，如 {1000} 或 {32, 64}
    dtype,      // 数据类型，如 torch::kFloat32
    spec,       // 离散值规格（TensorSpec）
    base_dist,  // 背景分布（BaseDist），可省略
    seed        // 可选的随机种子
);
```

---

## 第一个例子：纯背景分布

如果不需要任何特殊值，传一个空的 `TensorSpec{}`：

```cpp
// 1000 个元素，float32，全部来自 [-1, 1] 均匀分布（默认）
auto t = generate_tensor({1000}, torch::kFloat32, TensorSpec{});
```

可以换成其他背景分布：

```cpp
// 正态分布 N(0, 1)
auto t = generate_tensor({1000}, torch::kFloat32, TensorSpec{},
                         BaseDist::normal(0.0, 1.0));

// 整数均匀分布 [0, 256)
auto t = generate_tensor({1000}, torch::kInt32, TensorSpec{},
                         BaseDist::randint(0, 256));
```

---

## 注入 NaN

测试数值代码时，最常见的需求是在张量中混入 NaN，看代码能否正确处理。

### 10% 的元素是 NaN，其余是 [0, 1] 均匀分布

```cpp
auto t = generate_tensor({1000}, torch::kFloat32,
    TensorSpec{{SpecialValue::NaN, 0.1}},     // 10% NaN
    BaseDist::uniform(0.0, 1.0));
```

`TensorSpec{{SpecialValue::NaN, 0.1}}` 的含义：

- 花括号内的每一项是 `{值, 概率}`
- `SpecialValue::NaN` 表示静默 NaN（quiet NaN）
- `0.1` 表示该值占总元素的 10%
- 剩余 90% 由 `BaseDist` 填充

### 100% 全是 NaN

概率设为 1.0，此时 `BaseDist` 不起作用：

```cpp
auto t = generate_tensor({1000}, torch::kFloat32,
    TensorSpec{{SpecialValue::NaN, 1.0}});
```

### 便捷写法

对于单类特殊值，有专门的快捷函数：

```cpp
#include "convenience.h"

// 等价于上面的 10% NaN 示例
auto t = tensor_with_nan({1000}, torch::kFloat32, /*nan_ratio=*/0.1);
```

---

## 注入 Inf（无穷大）

### 5% 正无穷 + 5% 负无穷

```cpp
auto t = generate_tensor({1000}, torch::kFloat64,
    TensorSpec{
        {SpecialValue::Inf,  0.05},   // 5% +inf
        {SpecialValue::NInf, 0.05},   // 5% -inf
    },
    BaseDist::normal(0.0, 1.0));      // 剩余 90% 正态分布
```

### 同时有 NaN 和 Inf

多个特殊值可以自由组合，只要概率之和不超过 1.0：

```cpp
auto t = generate_tensor({10000}, torch::kFloat32,
    TensorSpec{
        {SpecialValue::NaN,  0.1},    // 10% NaN
        {SpecialValue::Inf,  0.05},   // 5% +inf
        {SpecialValue::NInf, 0.05},   // 5% -inf
    },
    BaseDist::uniform(-100.0, 100.0));
// 剩余 80% 来自 [-100, 100] 均匀分布
```

便捷写法：

```cpp
auto t = tensor_with_inf_nan({10000}, torch::kFloat32,
    /*inf_ratio=*/0.05, /*nan_ratio=*/0.1);
```

### 注意：Inf 不适用于所有 dtype

- **整数类型**（int8/16/32/64, uint8/16/32/64）：没有无穷大的概念，传入会报错
- **float8_e4m3fn**：该编码格式没有 inf 位模式，传入会报错
- **float8_e8m0fnu (E8M0)**：无符号指数格式，无穷大无效

```cpp
// 这行会抛出 std::invalid_argument
auto t = generate_tensor({100}, torch::kInt32,
    TensorSpec{{SpecialValue::Inf, 0.1}});
```

---

## 注入 DTypeMin / DTypeMax（类型极值）

`DTypeMin` 和 `DTypeMax` 是 dtype 感知的——同一个枚举在不同 dtype 下会自动映射为该类型的最小/最大可表示值：

| dtype | DTypeMin | DTypeMax |
|-------|----------|----------|
| float32 | -3.40e+38 (FLT_MAX 取反) | 3.40e+38 (FLT_MAX) |
| float16 | -65504.0 | 65504.0 |
| int8 | -128 | 127 |
| int32 | -2147483648 | 2147483647 |
| uint8 | 0 | 255 |

### 边界值测试：10% 最小值 + 10% 最大值

```cpp
auto t = generate_tensor({10000}, torch::kFloat32,
    TensorSpec{
        {SpecialValue::DTypeMin, 0.1},
        {SpecialValue::DTypeMax, 0.1},
    });
// 10% 元素 = -FLT_MAX，10% 元素 = FLT_MAX，80% 来自默认均匀分布
```

### 整数类型的边界测试

```cpp
auto t = generate_tensor({10000}, torch::kInt8,
    TensorSpec{
        {SpecialValue::DTypeMin, 0.2},   // 20% = -128
        {SpecialValue::DTypeMax, 0.2},   // 20% = 127
    },
    BaseDist::randint(-100, 100));       // 剩余 60% 来自 [-100, 100) 随机整数
```

便捷写法：

```cpp
auto t = tensor_with_max_min({10000}, torch::kInt8,
    /*max_ratio=*/0.2, /*min_ratio=*/0.2);
```

---

## 组合多种特殊值

可以在一个 TensorSpec 中组合任意数量的特殊值：

```cpp
// "压力测试"配置：NaN + Inf + 极值 + 零，各占 10%
auto t = generate_tensor({100000}, torch::kFloat32,
    TensorSpec{
        {SpecialValue::NaN,      0.1},
        {SpecialValue::Inf,      0.1},
        {SpecialValue::NInf,     0.1},
        {SpecialValue::DTypeMax, 0.1},
        {SpecialValue::DTypeMin, 0.1},
        {SpecialValue::Zero,     0.1},
    },
    BaseDist::normal(0.0, 1.0));
// 60% 特殊值 + 40% 正态分布
```

也可以把概率之和设为 1.0，生成一个只包含离散值的张量：

```cpp
// 纯离散：50% NaN + 25% 最大值 + 25% 最小值，没有背景分布
auto t = generate_tensor({1000}, torch::kFloat64,
    TensorSpec{
        {SpecialValue::NaN,      0.5},
        {SpecialValue::DTypeMax, 0.25},
        {SpecialValue::DTypeMin, 0.25},
    });
```

---

## 使用数值字面量

除了 `SpecialValue` 枚举，还可以直接用数字作为离散值：

```cpp
// 30% 填充值 42，20% 填充值 -1，其余均匀分布
auto t = generate_tensor({1000}, torch::kInt32,
    TensorSpec{{42, 0.3}, {-1, 0.2}},
    BaseDist::randint(0, 100));

// 混合：20% NaN + 30% 固定值 99.5，其余正态分布
auto t = generate_tensor({1000}, torch::kFloat32,
    TensorSpec{{SpecialValue::NaN, 0.2}, {99.5, 0.3}},
    BaseDist::normal(0.0, 1.0));
```

**隐式提升：** 某些数字会自动等价于对应的 SpecialValue：

```cpp
// 以下两行完全等价：
TensorSpec{{0, 0.3}}                      // 0 → SpecialValue::Zero
TensorSpec{{SpecialValue::Zero, 0.3}}

// 同样等价的还有：1 → One，-1 → NOne
```

这意味着 `-1` 在 uint8 上会被填充为 `255`（dtype 感知行为），而不是截断为 0。如果你只想填入原始数值 `-1` 并让 dtype 自然 cast，可以用 `-1.0`（double 字面量且不等于整数 -1 的提升路径——但实际上 `-1.0 == -1.0` 也会被提升）。对于 `-1` 这种情况，SpecialValue 行为通常正是你需要的。

---

## 背景分布一览

`BaseDist` 控制非离散部分的填充方式。所有工厂方法都支持可选的 `seed` 末参数：

```cpp
BaseDist::uniform(lo, hi)               // 均匀分布 [lo, hi]
BaseDist::normal(mean, std)             // 正态分布
BaseDist::randint(lo, hi)               // 整数均匀 [lo, hi)，仅限整数 dtype
BaseDist::exponential(lambda)           // 指数分布
BaseDist::poisson(rate)                 // 泊松分布，仅限整数 dtype
BaseDist::lognormal(mean, std)          // 对数正态分布
BaseDist::cauchy(median, sigma)         // 柯西分布（产生极端值）
BaseDist::bernoulli(p)                  // 伯努利分布（0 或 1）

// 带 seed 的版本（创建隔离的 Generator，跨调用自动推进）：
BaseDist::uniform(0.0, 1.0, /*seed=*/42)
```

如果省略 `BaseDist`，默认使用 `Uniform(-1, 1)`。

---

## 便捷接口速查

对于常见的单类/双类特殊值场景，`convenience.h` 提供了命名函数，省去手写 TensorSpec：

```cpp
#include "convenience.h"

// ---- 单类特殊值 ----
tensor_with_nan  (shape, dtype, ratio)    // 注入 NaN
tensor_with_inf  (shape, dtype, ratio)    // 注入 +Inf
tensor_with_ninf (shape, dtype, ratio)    // 注入 -Inf
tensor_with_min  (shape, dtype, ratio)    // 注入 DTypeMin
tensor_with_max  (shape, dtype, ratio)    // 注入 DTypeMax
tensor_with_zero (shape, dtype, ratio)    // 注入 Zero
tensor_with_one  (shape, dtype, ratio)    // 注入 One
tensor_with_none (shape, dtype, ratio)    // 注入 -1
tensor_with_epsilon(shape, dtype, ratio)  // 注入 Epsilon
tensor_with_tiny (shape, dtype, ratio)    // 注入 Tiny

// ---- 双类特殊值（45 个组合） ----
tensor_with_inf_nan (shape, dtype, inf_ratio, nan_ratio)
tensor_with_max_min (shape, dtype, max_ratio, min_ratio)
tensor_with_nan_zero(shape, dtype, nan_ratio, zero_ratio)
// ... 全部 C(10,2) = 45 对组合均可用
```

所有便捷函数默认 `dtype = kFloat32`，`ratio = 0.01`，背景分布为 `Uniform(0, 1)`。

---

## 策略生成器：批量生成测试配置

当你需要系统化地测试大量特殊值组合时，手动写 TensorSpec 很繁琐。策略生成器自动产出 `TestCase` 列表，每个 `TestCase` 包含一个 `TensorSpec` 和一个 `BaseDist`，可直接用于 `generate_tensor`。

```cpp
#include "strategies.h"

struct TestCase {
    TensorSpec spec;     // 离散值配置
    BaseDist dist;       // 背景分布
    std::string label;   // 可读标签，如 "NaN=30%+Inf=20%"
};
```

### 策略 1：全随机

随机选取特殊值种类和比例，用于发现意料之外的边界组合：

```cpp
auto cases = random_strategy(
    50,                  // 生成 50 组配置
    torch::kFloat32,     // 针对 float32（自动过滤无效的特殊值）
    /*p_max=*/0.5);      // 离散值总概率上限 50%

for (auto& tc : cases) {
    std::cout << tc.label << std::endl;
    auto tensor = generate_tensor({1000}, torch::kFloat32, tc.spec, tc.dist);
    // ... 用 tensor 做测试
}
```

### 策略 2：固定类别

指定只用 NaN、Inf、DTypeMax 这三种，但比例随机：

```cpp
auto cases = fixed_categories_strategy(
    {SpecialValue::NaN, SpecialValue::Inf, SpecialValue::DTypeMax},
    20,                  // 20 组配置
    torch::kFloat32);

for (auto& tc : cases) {
    auto tensor = generate_tensor({1000}, torch::kFloat32, tc.spec, tc.dist);
}
```

### 策略 3：单类/双类策略

精确控制某一种（或两种）特殊值的比例：

```cpp
// 50 组配置，每组都有 NaN，比例随机在 (0, 30%] 之间
auto cases = single_sv_strategy(SpecialValue::NaN, 50, torch::kFloat32,
                                /*p=*/0.3);

// 30 组配置，NaN + Inf 双类，NaN 最多 20%，Inf 最多 10%
auto cases = dual_sv_strategy(
    SpecialValue::NaN, SpecialValue::Inf,
    30, torch::kFloat32,
    /*p0=*/0.2, /*p1=*/0.1);
```

### 策略 4：网格覆盖

适合做确定性的、系统化的覆盖测试：

```cpp
// 4a: 每种有效特殊值各一个 100% 纯填充的 TestCase
auto cases = grid_single_value(torch::kFloat32);
// float32 有 10 种有效特殊值 → 产出 10 组

// 4b: 所有特殊值按 10% 粒度混合
auto cases = grid_full_mix(torch::kFloat32, /*min_count=*/50);
// 至少 50 组唯一配置

// 4c: 每对特殊值的 5 档网格：(100%,0%), (75%,25%), (50%,50%), (25%,75%), (0%,100%)
auto cases = grid_pairwise(torch::kFloat32);
// C(10,2) × 5 = 225 组
```

### 遍历生成

所有策略的输出都是 `std::vector<TestCase>`，使用方式统一：

```cpp
auto cases = grid_pairwise(torch::kFloat32);
for (auto& tc : cases) {
    auto tensor = generate_tensor({10000}, torch::kFloat32, tc.spec, tc.dist);
    // 用 tensor 跑你的被测函数，检查结果
    std::cout << "Testing: " << tc.label << std::endl;
}
```

你也可以替换背景分布：

```cpp
for (auto& tc : cases) {
    tc.dist = BaseDist::normal(0.0, 1.0);  // 用正态分布替代默认均匀分布
    auto tensor = generate_tensor({10000}, torch::kFloat32, tc.spec, tc.dist);
}
```

---

## 可复现性控制

### 场景 A：固定生成某一个特定张量

每次调用都得到完全相同的张量，适合调试时锁定输入：

```cpp
auto t = generate_tensor({1000}, torch::kFloat32,
    TensorSpec{{SpecialValue::NaN, 0.1}},
    BaseDist::uniform(0.0, 1.0),
    /*seed=*/42);
// 无论调用多少次，t 始终相同
```

### 场景 B：生成确定性序列

多次调用得到不同但可复现的张量序列，适合回归测试：

```cpp
auto dist = BaseDist::uniform(0.0, 1.0, /*seed=*/42);
auto spec = TensorSpec{{SpecialValue::NaN, 0.1}, {SpecialValue::Inf, 0.05}};

auto t1 = generate_tensor({1000}, torch::kFloat32, spec, dist);
auto t2 = generate_tensor({1000}, torch::kFloat32, spec, dist);
auto t3 = generate_tensor({1000}, torch::kFloat32, spec, dist);
// t1 ≠ t2 ≠ t3，但每次运行程序时这三个张量的内容完全一致
```

### 场景 C：不关心可复现性

省略 seed 和 Generator，让全局 RNG 自由推进：

```cpp
auto t = generate_tensor({1000}, torch::kFloat32,
    TensorSpec{{SpecialValue::NaN, 0.1}});
// 每次运行结果不同
```

---

## 完整示例：测试一个求和函数

假设你要测试一个自定义的 `safe_sum` 函数，需要验证它在各种边界输入下的表现：

```cpp
#include "tensor_generator.h"
#include "convenience.h"
#include "strategies.h"

void test_safe_sum() {
    // 1. 基本正确性：普通数值
    auto normal_input = generate_tensor({1000}, torch::kFloat32, TensorSpec{},
                                        BaseDist::normal(0.0, 1.0, /*seed=*/1));
    auto result = safe_sum(normal_input);
    // assert(result 是有限数 && 接近 0)

    // 2. NaN 处理：结果应该是 NaN 或被跳过
    auto nan_input = tensor_with_nan({1000}, torch::kFloat32, /*ratio=*/0.1);
    auto result_nan = safe_sum(nan_input);
    // assert(结果符合你的 NaN 策略)

    // 3. Inf 处理：正负无穷相消？
    auto inf_input = generate_tensor({1000}, torch::kFloat32,
        TensorSpec{
            {SpecialValue::Inf,  0.1},
            {SpecialValue::NInf, 0.1},
        },
        BaseDist::uniform(-1.0, 1.0, /*seed=*/2));
    auto result_inf = safe_sum(inf_input);

    // 4. 极值溢出：大量 DTypeMax 相加
    auto overflow_input = generate_tensor({1000}, torch::kFloat32,
        TensorSpec{{SpecialValue::DTypeMax, 0.5}},
        BaseDist::uniform(0.0, 1.0, /*seed=*/3));
    auto result_overflow = safe_sum(overflow_input);
    // assert(结果是 Inf 或被正确处理)

    // 5. 系统化覆盖：用策略生成器遍历所有组合
    auto cases = grid_pairwise(torch::kFloat32);
    for (auto& tc : cases) {
        auto input = generate_tensor({10000}, torch::kFloat32, tc.spec, tc.dist);
        auto result = safe_sum(input);
        // assert(结果不崩溃、没有未定义行为)
    }
}
```

---

## 常见错误与排查

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `discrete value not valid for given dtype` | 特殊值与 dtype 不兼容 | 检查 SpecialValue 和 dtype 的组合，例如整数不支持 NaN |
| `probability must be non-negative` | 概率为负数 | 检查 TensorSpec 中的概率值 |
| `discrete probabilities exceed 1.0` | 概率之和超过 1 | 减小各项概率，使总和 <= 1.0 |
| `RandInt base distribution is incompatible with float dtype` | 用 RandInt 配浮点类型 | 改用 Uniform 或 Normal |
| `raw value out of int64 range` | 数值字面量超出 int64 范围 | 缩小数值到 int64 可表示范围 |
