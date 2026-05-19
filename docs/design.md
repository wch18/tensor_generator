# tensor_generator 设计文档

## 架构概览

本库分为五个模块，各司其职：

```
src/
  special_value.h / special_value.cpp   — dtype 感知的特殊值表
  tensor_spec.h                         — 声明式分布规格（仅头文件）
  tensor_generator.h / tensor_generator.cpp — 生成引擎（公开 API）
  convenience.h                         — 便捷封装接口（仅头文件）
  strategies.h / strategies.cpp         — 策略生成器
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `special_value` | 将 `SpecialValue` 枚举成员映射为每种 dtype 对应的精确数值表示；校验 dtype/值的兼容性 |
| `tensor_spec` | 将离散 `(SpecialValue, probability)` 对列表与可选的连续基础分布聚合为单一 `TensorSpec` 结构体 |
| `tensor_generator` | 实现生成算法；暴露 `generate_tensor()` |
| `convenience` | 为常用模式提供一行调用的便捷封装函数 |
| `strategies` | 生成 `TestCase` 向量，用于系统化测试覆盖 |

依赖图为严格的分层结构：

```
strategies → convenience → tensor_generator → tensor_spec → special_value → <torch/torch.h>
```

无循环依赖。

---

## 公开 API

```cpp
torch::Tensor generate_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype,
    const TensorSpec& spec,
    const BaseDist& base_dist = {},            // 默认：Uniform(-1, 1)
    std::optional<int64_t> seed = std::nullopt);
```

`TensorSpec` 仅包含离散条目。`BaseDist` 是控制连续基础分布的独立参数。`seed` 提供可复现性，无需调用方管理全局 RNG 状态。

---

## "基础层 + 掩码覆盖"算法

`generate_tensor(shape, dtype, spec, base_dist, seed)` 分三个阶段工作：

### 阶段 1 — 基础张量

令 `p_sum = 所有离散条目概率之和`。

- 若 `p_sum >= 1.0 - 1e-9`（全离散模式）：分配一个零初始化的 `dtype` 张量（通过安全的中间类型——见下文"类型转换解决方案"）。
- 否则：使用 `BaseDist`（均匀 `U(lo, hi)` 或正态 `N(mu, sigma)`）在 `float64` 精度下采样连续基础张量，然后转换为 `dtype`。

### 阶段 2 — 选择器张量

生成一个与 `shape` 相同的 `float64` 均匀张量 `selector`，范围为 `[0, 1)`。它独立于基础张量，在所有离散条目间复用。

### 阶段 3 — 掩码覆盖

按顺序遍历离散条目，维护累积概率 `cum`：

```
for each entry (value, p) in spec.discrete:
    if p == 0: 跳过
    if 这是最后一个非零条目 AND 全离散模式:
        mask = (selector >= cum)          # 吸收舍入余量
    else:
        mask = (selector >= cum) AND (selector < cum + p)
    masked_fill_(result, mask, value)
    cum += p
```

最后一个条目的特殊处理确保：当概率以小数指定（如 `0.1 + 0.2 + 0.3 + 0.4`）且浮点求和可能略大或略小于 `1.0` 时，不会有元素意外保留在初始值。

---

## SpecialValue 值表

下表展示每种 dtype 下各 `SpecialValue` 枚举成员写入张量的具体值。"二补数回绕"指有符号域中 `-1` 的位模式，被重新解释为无符号类型。

| SpecialValue | uint8 | uint16 | uint32 | uint64 | int8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 | bool | fp8_e4m3fn | fp8_e5m2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Zero | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | false | 0.0 | 0.0 |
| One | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | true | 1.0 | 1.0 |
| NegativeOne | 255 | 65535 | 4294967295 | 2^64-1 | -1 | -1 | -1 | -1 | -1.0 | -1.0 | -1.0 | -1.0 | true | -1.0 | -1.0 |
| DTypeMin | 0 | 0 | 0 | 0 | -128 | -32768 | INT32_MIN | INT64_MIN | -65504.0 | -3.39e+38 | -FLT_MAX | -DBL_MAX | false | -448.0 | -57344.0 |
| DTypeMax | 255 | 65535 | 4294967295 | 2^64-1 | 127 | 32767 | INT32_MAX | INT64_MAX | 65504.0 | 3.39e+38 | FLT_MAX | DBL_MAX | true | 448.0 | 57344.0 |
| NaN | — | — | — | — | — | — | — | — | NaN | NaN | NaN | NaN | — | NaN | NaN |
| Infinity | — | — | — | — | — | — | — | — | +inf | +inf | +inf | +inf | — | — | +inf |
| NegativeInfinity | — | — | — | — | — | — | — | — | -inf | -inf | -inf | -inf | — | — | -inf |
| Epsilon | — | — | — | — | — | — | — | — | 9.77e-4 | 7.81e-3 | FLT_EPSILON | DBL_EPSILON | — | 0.125 | 0.25 |
| Tiny | — | — | — | — | — | — | — | — | 6.10e-5 | FLT_MIN | FLT_MIN | DBL_MIN | — | 0.015625 | 6.10e-5 |

"—"表示该值对该 dtype 无效（`is_valid_for_dtype()` 返回 `false`）。

---

## 边界情况处理

### 全离散规格中的浮点累加问题

当所有离散概率理论上恰好求和为 `1.0` 时，其浮点表示的实际和可能略大或略小于 `1.0`。朴素的范围检查 `selector < cum` 可能留下一小段选择器范围未被分配。修复方法是对最后一个非零条目使用 `selector >= cum`（无上界），无条件吸收舍入余量。

### uint64 二补数

PyTorch 不直接支持对 `uint64` 张量执行 `masked_fill_`。解决方案是将结果张量转换为 `int64`，通过 `masked_fill_` 填充（int64 的位模式因二补数而完全相同），然后转回 `uint64`。这正确地将 `NegativeOne` 表示为 `2^64 - 1`，`DTypeMax` 也是同一值。

### FP8 (float8_e4m3fn / float8_e5m2) 类型转换解决方案

PyTorch 的 `masked_fill_` 在目标 dtype 为 FP8 时不接受标量。解决方案是将张量提升为 `float64`，执行填充，然后转回。由于为 FP8 选择的所有特殊值在 `float64` 中均可无损表示，因此保留了精确的 FP8 可表示值。

### float8_e4m3fn 无无穷大

`e4m3fn` 编码将本应表示无穷大的位模式保留给了额外的 NaN 值。因此 `Infinity` 和 `NegativeInfinity` 被 `is_valid_for_dtype()` 对该 dtype 显式拒绝。

### 布尔类型

`bool` 被视为 1 位无符号整数。`DTypeMin` = `false` (0)，`DTypeMax` = `true` (1)。`NegativeOne` 映射为 `true`（通过二补数规则与 `One` 相同）。所有浮点专用值（NaN、Inf、Epsilon、Tiny）均无效。

---

## 测试策略

测试分为两个层次：

### 1. 参数化笛卡尔积套件（`CartesianTest`）

该套件实例化 `shape × dtype × config` 的组合。不兼容的组合通过 `GTEST_SKIP()` 跳过（使用每配置的兼容性谓词），使注册的总测试数保持可控。

| 维度 | 数量 | 取值 |
|------|------|------|
| 形状 | 8 | 一维（大小 1、7、100、1M）、二维、三维、四维、五维 |
| Dtype | 15 | 所有支持的 dtype（见下表） |
| 配置 | 16 | 各种概率分布 |

每组合的验证策略：
- 输出的形状和 dtype 被精确断言。
- 对于 `N >= 10,000`：观测到的离散值计数与预期计数通过 5-sigma 高斯容差进行比较（对非常稀有的值，最小松弛量为 3 个计数）。
- 对于全离散规格（`p_sum ≈ 1.0`）：断言所有离散值的总计数等于 `N`（泄漏检查）。
- 低精度 dtype（整数、FP8、bool）带连续分量时使用宽松的下界检查，因为量化可能导致连续样本与离散值重合。

### 2. 专项单元测试（非参数化）

覆盖笛卡尔扫描仅能统计性验证的特定行为：

| 分组 | 测试内容 |
|------|----------|
| `Validation` | `is_valid_for_dtype()` 对浮点专用和 e4m3fn 特有规则的验证 |
| `ErrorHandling` | 对整数使用 NaN、对 e4m3fn 使用 inf、负概率、概率总和过大时抛出 `std::invalid_argument` |
| `EdgeCase` | 空张量、单元素、零概率条目、极小概率（p=0.001 在 N=1M 时） |
| `UInt{8,16,32,64}Specific` | 精确的二补数值、无泄漏检查 |
| `FloatSpecials` | 每种浮点 dtype 的精确 epsilon 和 tiny 值 |
| `IntBoundary` | int8/16/32/64 的精确最大/最小值 |
| `BoolDType` | Zero=false、One=true、DTypeMin/Max |
| `Performance` | 400 万元素张量；统计检查 |
| `ContinuousDist` | 均匀分布的范围边界和均值；正态分布的均值和标准差 |
| `AllDiscrete` | 50 轮泄漏压力测试，验证浮点累加修复 |

---

## 便捷接口（`convenience.h`）

仅头文件模块，为常用张量生成模式提供简写函数。所有函数默认使用 `Uniform(0, 1)` 作为基础分布。

### 基础分布生成器

```cpp
make_uniform_tensor(shape, dtype=kFloat32, lo=0.0, hi=1.0, generator=nullopt)
make_normal_tensor(shape, dtype=kFloat32, mean=0.0, std=1.0, generator=nullopt)
make_randint_tensor(shape, dtype=kInt32, lo=0, hi=100, generator=nullopt)
```

### 单类 SpecialValue 函数

```cpp
tensor_with_one_sv(sv, shape, dtype=kFloat32, ratio=0.01, generator=nullopt)  // 通用
tensor_with_nan(shape, dtype=kFloat32, nan_ratio=0.01, generator=nullopt)     // 命名版
// 10 个命名变体：zero, one, none, min, max, nan, inf, ninf, epsilon, tiny
```

### 双类 SpecialValue 函数

```cpp
tensor_with_two_sv(sv1, ratio1, sv2, ratio2, shape, dtype=kFloat32, generator=nullopt)  // 通用
tensor_with_inf_nan(shape, dtype=kFloat32, inf_ratio=0.01, nan_ratio=0.01, generator=nullopt)
// 45 个命名变体（全部 C(10,2) 对），通过 X-macro 生成
```

命名规则：短名按字母序排列（`epsilon` < `inf` < `max` < `min` < `nan` < `ninf` < `none` < `one` < `tiny` < `zero`）。

---

## 策略生成器（`strategies.h/.cpp`）

生成 `std::vector<TestCase>` 用于系统化的张量生成循环。

```cpp
struct TestCase {
    TensorSpec spec;    // 离散条目
    BaseDist dist;      // 基础分布（默认 Uniform(0,1)）
    std::string label;  // 可读标识符
};
```

### 2a — 全随机策略

随机选取 SpecialValue 类型和概率，按 dtype 有效性过滤。

| 参数 | 描述 |
|------|------|
| `count` | 生成的 TestCase 数量 |
| `dtype` | 过滤哪些 SpecialValue 有效 |
| `p_max` | 离散概率总和上限（默认 1.0） |
| `strict` | 若为 true，总和 == p_max；否则总和 ∈ (0, p_max] |
| `generator` | 可选的 Generator 用于可复现性 |

算法：随机选取 k 个有效 SV，通过 Dirichlet(1,...,1) 缩放至目标值分配概率。

### 2b — 固定类别策略

与全随机相同，但用户指定使用哪些 SpecialValue。仅比例被随机化。

### 2c — 单类/双类策略

带显式概率目标的命名封装。

- `single_sv_strategy(sv, count, dtype, p=0.5, strict=false, gen)` — 一种 SV，比例在 (0, p] 内（strict 时恰好为 p）
- `dual_sv_strategy(sv1, sv2, count, dtype, p0=0.3, p1=0.3, strict0=false, strict1=false, gen)` — 两种 SV 带独立比例约束，钳制使总和 ≤ 1.0；若双方都为 strict 且 p0+p1 > 1.0 则抛出异常

### 2d — 网格策略

确定性或半确定性的覆盖模式。

**2d-i：单一极值网格**（`grid_single_value(dtype)`）

每个有效 SpecialValue 以 100% 概率生成一个 TestCase。生成 N 个用例，N = dtype 的有效 SV 数量（如 float32 为 10，int32 为 5）。

**2d-ii：全混合网格**（`grid_full_mix(dtype, min_count=50, gen)`）

所有有效 SV + "正常值"以 10% 网格粒度混合。每个比例为 10% 的整数倍，所有比例之和为 100%。通过将 10 个"份额"随机分配到 k 个类别来采样唯一的网格对齐组合。保证 ≥ min_count 个唯一配置。

**2d-iii：两两组合网格**（`grid_pairwise(dtype)`）

对每对有效 SV 的无序组合，生成 5 个配置，对应 25% 网格点：
(100%, 0%)、(75%, 25%)、(50%, 50%)、(25%, 75%)、(0%, 100%)。

总用例数 = C(N, 2) × 5，其中 N = 有效 SV 数量（如 float32 为 225，int32 为 50）。

---

## 支持的 Dtype 表

| dtype | PyTorch 常量 | 位宽 | 有符号 | 浮点 |
|-------|-------------|------|--------|------|
| uint8 | `torch::kUInt8` | 8 | 否 | 否 |
| uint16 | `c10::ScalarType::UInt16` | 16 | 否 | 否 |
| uint32 | `c10::ScalarType::UInt32` | 32 | 否 | 否 |
| uint64 | `c10::ScalarType::UInt64` | 64 | 否 | 否 |
| int8 | `torch::kInt8` | 8 | 是 | 否 |
| int16 | `torch::kInt16` | 16 | 是 | 否 |
| int32 | `torch::kInt32` | 32 | 是 | 否 |
| int64 | `torch::kInt64` | 64 | 是 | 否 |
| float16 | `torch::kFloat16` | 16 | 是 | 是 |
| bfloat16 | `torch::kBFloat16` | 16 | 是 | 是 |
| float32 | `torch::kFloat32` | 32 | 是 | 是 |
| float64 | `torch::kFloat64` | 64 | 是 | 是 |
| bool | `torch::kBool` | 1 | 否 | 否 |
| float8_e4m3fn | `torch::kFloat8_e4m3fn` | 8 | 是 | 是（无 inf） |
| float8_e5m2 | `torch::kFloat8_e5m2` | 8 | 是 | 是 |
