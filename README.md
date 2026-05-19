# tensor_generator

一个用于生成具有受控分布的 libtorch 张量的 C++ 库——将离散特殊值（NaN、Inf、dtype 最大/最小值等）与连续/整数基础分布（uniform/normal/randint）组合使用。

专为测试需要处理各种 PyTorch 标量类型边界情况的数值代码而设计。

## 特性

- **支持 16 种 dtype**：uint8/16/32/64、int8/16/32/64、float16/bfloat16/float32/float64、bool、float8_e4m3fn、float8_e5m2、float8_e8m0fnu
- **离散特殊值**：Zero、One、NOne、DTypeMin、DTypeMax、NaN、Inf、NInf、Epsilon、Tiny
- **任意数值字面量**：支持 `{42, 0.3}` 直接指定离散值，无需使用 SpecialValue 枚举
- **基础分布**：Uniform、Normal、RandInt、Exponential、Poisson、LogNormal、Cauchy、Bernoulli
- **概率混合**：以指定概率将离散值与连续/整数背景分布组合
- **便捷接口**：常用模式的一行调用函数（`tensor_with_nan`、`tensor_with_inf_nan` 等）
- **策略生成器**：为系统化覆盖生成测试用例配置（随机、网格、两两组合）
- **torch::Generator 支持**：BaseDist 接受 Generator 以实现隔离的随机数状态；支持从 Generator 隐式转换
- **可复现**：可选的 seed 参数或每分布独立 Generator
- **dtype 感知校验**：运行时拒绝无效组合（如整数类型使用 NaN、浮点类型使用 RandInt）

## 依赖要求

| 依赖项 | 版本 |
|--------|------|
| PyTorch (libtorch) | >= 2.9 |
| C++ 标准 | C++17 |
| CMake | >= 3.18 |
| GoogleTest | （仅测试需要） |

## 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
make -j$(nproc)
```

运行测试：

```bash
./tensor_generator_test              # 核心测试（1700+）
./test_convenience_strategies        # 便捷接口与策略测试（60）
```

## 快速开始

```cpp
#include "tensor_generator.h"

// 纯均匀分布 [-1, 1]（默认）
auto t1 = generate_tensor({1000}, torch::kFloat32, TensorSpec{});

// 30% 零值，其余为 [-5, 5] 均匀分布
auto t2 = generate_tensor({1000}, torch::kFloat32,
    TensorSpec{{SpecialValue::Zero, 0.3}},
    BaseDist::uniform(-5.0, 5.0));

// 20% NaN + 10% Inf，其余为正态分布 N(0, 1)
auto t3 = generate_tensor({1000}, torch::kFloat64,
    TensorSpec{{SpecialValue::NaN, 0.2}, {SpecialValue::Inf, 0.1}},
    BaseDist::normal(0.0, 1.0));

// 使用数值字面量：50% 填 42，其余为默认均匀分布
auto t4 = generate_tensor({1000}, torch::kInt32, TensorSpec{{42, 0.5}});

// 混合：20% NaN + 30% 值为 99，其余为正态分布
auto t5 = generate_tensor({1000}, torch::kFloat32,
    TensorSpec{{SpecialValue::NaN, 0.2}, {99, 0.3}},
    BaseDist::normal(0.0, 1.0));
```

## 便捷接口

```cpp
#include "convenience.h"

// --- 基础分布生成器 ---
auto t1 = make_uniform_tensor({1000}, torch::kFloat32, 0.0, 1.0);
auto t2 = make_normal_tensor({1000}, torch::kFloat64, 0.0, 1.0);
auto t3 = make_randint_tensor({1000}, torch::kInt32, -100, 100);
auto t4 = make_exponential_tensor({1000}, torch::kFloat32, /*lambda=*/2.0);
auto t5 = make_poisson_tensor({1000}, torch::kInt32, /*rate=*/5.0);
auto t6 = make_lognormal_tensor({1000}, torch::kFloat32, /*mean=*/0.0, /*std=*/1.0);
auto t7 = make_cauchy_tensor({1000}, torch::kFloat32, /*median=*/0.0, /*sigma=*/1.0);
auto t8 = make_bernoulli_tensor({1000}, torch::kFloat32, /*p=*/0.3);

// --- 单类特殊值（10 个命名函数） ---
auto t9 = tensor_with_nan({1000}, torch::kFloat32, 0.1);
auto t10 = tensor_with_inf({1000}, torch::kFloat32, 0.05);
auto t11 = tensor_with_zero({1000}, torch::kFloat32, 0.2);
// 还有：tensor_with_one, tensor_with_none, tensor_with_min, tensor_with_max,
//       tensor_with_ninf, tensor_with_epsilon, tensor_with_tiny

// --- 通用单类特殊值 ---
auto t12 = tensor_with_one_sv(SpecialValue::NaN, {1000}, torch::kFloat32, 0.1);

// --- 双类特殊值（45 个命名函数） ---
auto t13 = tensor_with_inf_nan({1000}, torch::kFloat32, 0.1, 0.05);
auto t14 = tensor_with_max_min({1000}, torch::kInt32, 0.2, 0.2);
// 全部 C(10,2) = 45 对组合均可用

// --- 通用双类特殊值 ---
auto t15 = tensor_with_two_sv(
    SpecialValue::NaN, 0.1, SpecialValue::Inf, 0.05,
    {1000}, torch::kFloat32);

// --- 使用 Generator 确保可复现 ---
auto gen = at::detail::createCPUGenerator();
gen.set_current_seed(42);
auto t16 = tensor_with_nan({1000}, torch::kFloat32, 0.1, gen);
```

## 策略生成器

```cpp
#include "strategies.h"

// 策略生成器输出 TestCase 向量，用于系统化遍历生成：
//   struct TestCase { TensorSpec spec; BaseDist dist; std::string label; };

// --- 2a：全随机（类型和比例均随机化） ---
auto cases = random_strategy(50, torch::kFloat32, /*p_max=*/0.5, /*strict=*/false);

// --- 2b：固定类别（用户指定特殊值种类，比例随机化） ---
auto cases = fixed_categories_strategy(
    {SpecialValue::NaN, SpecialValue::Inf, SpecialValue::Zero},
    20, torch::kFloat32, /*p_max=*/0.8, /*strict=*/true);

// --- 2c：单类/双类，带目标概率 ---
auto cases = single_sv_strategy(SpecialValue::NaN, 10, torch::kFloat32,
    /*p=*/0.3, /*strict=*/true);
auto cases = dual_sv_strategy(SpecialValue::NaN, SpecialValue::Inf,
    10, torch::kFloat32,
    /*p0=*/0.2, /*p1=*/0.1, /*strict0=*/false, /*strict1=*/true);

// --- 2d：网格策略 ---
auto single = grid_single_value(torch::kFloat32);       // 10 组（每种 SV 各一，100%）
auto mixed  = grid_full_mix(torch::kFloat32, 50);       // >=50 组唯一的 10% 网格配置
auto pairs  = grid_pairwise(torch::kFloat32);           // 225 组（45 对 × 5 个网格点）

// --- 遍历生成张量（可替换 BaseDist） ---
for (auto& tc : cases) {
    tc.dist = BaseDist::exponential(1.0);  // 覆盖底噪为指数分布
    auto tensor = generate_tensor({1000}, torch::kFloat32, tc.spec, tc.dist);
}
```

**策略一览：**

| 策略 | 主要参数 | 含义 |
|------|----------|------|
| `random_strategy` | `count`, `dtype`, `p_max`, `strict` | 随机选取特殊值种类和比例，生成 count 组配置；离散总概率 ≤ p_max |
| `fixed_categories_strategy` | `categories`, `count`, `dtype`, `p_max`, `strict` | 固定特殊值种类，仅随机分配各自比例；总概率 ≤ p_max |
| `single_sv_strategy` | `sv`, `count`, `dtype`, `p`, `strict` | 单一特殊值，比例 ∈ (0, p]（strict=true 时固定为 p） |
| `dual_sv_strategy` | `sv1`, `sv2`, `count`, `dtype`, `p0`, `p1`, `strict0`, `strict1` | 两种特殊值各自独立控制比例上限 |
| `grid_single_value` | `dtype` | 每种有效特殊值各生成一个 100% 纯填充配置 |
| `grid_full_mix` | `dtype`, `min_count` | 所有特殊值 + 底噪，以 10% 粒度随机组合，去重后至少 min_count 个 |
| `grid_pairwise` | `dtype` | 两两配对，按 (100,0)/(75,25)/(50,50)/(25,75)/(0,100) 五档网格展开 |

## 核心 API

```cpp
torch::Tensor generate_tensor(
    torch::IntArrayRef shape,
    torch::ScalarType dtype,
    const TensorSpec& spec,
    const BaseDist& base_dist = {},              // 默认：Uniform(-1, 1)
    std::optional<int64_t> seed = std::nullopt);
```

**BaseDist 构造方式：**

```cpp
BaseDist{}                              // Uniform(-1, 1)
BaseDist::uniform(lo, hi)               // 均匀分布 [lo, hi]
BaseDist::uniform(lo, hi, seed)         // 均匀分布 + 隔离 Generator
BaseDist::normal(mean, std)             // 正态分布 N(mean, std)
BaseDist::normal(mean, std, seed)       // 正态分布 + 隔离 Generator
BaseDist::randint(lo, hi)               // 整数均匀分布 [lo, hi)
BaseDist::randint(lo, hi, seed)         // 整数均匀分布 + 隔离 Generator
BaseDist::exponential(lambda)           // 指数分布 Exp(lambda)
BaseDist::poisson(rate)                 // 泊松分布 Poisson(rate)
BaseDist::lognormal(mean, std)          // 对数正态分布 LogN(mean, std)
BaseDist::cauchy(median, sigma)         // 柯西分布 Cauchy(median, sigma)
BaseDist::bernoulli(p)                  // 伯努利分布 Bernoulli(p)
// 以上所有工厂方法均可附加可选的末参数 seed（int64_t）
BaseDist(torch::Generator gen)          // 隐式转换：Uniform(-1,1) + generator
dist.with_generator(gen)                // 为任意分布绑定 generator
dist.with_seed(seed)                    // 创建隔离 Generator 并绑定
```

**Dtype/BaseDist 兼容性：**

| BaseDist | 浮点类型 | 整数类型 | Bool |
|----------|:---:|:---:|:---:|
| Uniform | 可用 | 可用（截断） | 可用 |
| Normal | 可用 | 可用（截断） | 可用 |
| RandInt | 报错 | 可用 | 报错 |
| Exponential | 可用 | 可用（截断） | 可用 |
| Poisson | 报错 | 可用 | 报错 |
| LogNormal | 可用 | 可用（截断） | 可用 |
| Cauchy | 可用 | 可用（截断） | 可用 |
| Bernoulli | 可用 | 可用（截断） | 可用 |

**SpecialValue 枚举：**

| 名称 | 描述 |
|------|------|
| `Zero` | 加法单位元（E8M0 不可用） |
| `One` | 乘法单位元 |
| `NOne` | -1（无符号类型为 0xFF...FF；E8M0 不可用） |
| `DTypeMin` | 类型可表示的最小值 |
| `DTypeMax` | 类型可表示的最大值 |
| `NaN` | 静默 NaN（仅浮点类型） |
| `Inf` | +无穷大（浮点类型，不含 e4m3fn/e8m0fnu） |
| `NInf` | -无穷大（浮点类型，不含 e4m3fn/e8m0fnu） |
| `Epsilon` | 机器精度（仅浮点类型） |
| `Tiny` | 最小正规化数（仅浮点类型） |

## 项目结构

```
src/
  tensor_generator.h/.cpp  — 公开 API（generate_tensor）
  tensor_spec.h            — TensorSpec、BaseDist、DiscreteValue、DiscreteEntry
  special_value.h/.cpp     — SpecialValue 枚举与 dtype 感知的值表
  dtype_traits.h           — 内部 dtype 分类辅助函数
  convenience.h            — 便捷封装接口（仅头文件）
  strategies.h/.cpp        — 策略生成器（TestCase、随机/网格/两两组合）
test/
  tensor_generator_test.cpp         — 1700+ 核心测试用例
  test_configs.h                    — 共享测试配置
  test_convenience_strategies.cpp   — 60 个便捷接口与策略测试
  test_rng_trajectory.cpp           — RNG 轨迹确定性验证（独立程序，非 gtest）
docs/
  design.md                — 详细设计文档
```

## 附录：Seed 设置与 RNG 行为

有三种方式控制随机数生成的确定性，对应不同的使用场景：

| 设置方式 | 跨调用行为 | 跨运行行为 | 适用场景 |
|----------|------------|------------|----------|
| `generate_tensor(..., seed=42)` | 每次调用产生**相同**张量 | 相同 | 需要固定某一个张量内容 |
| 不传 seed，不用 Generator | 每次调用产生**不同**张量 | 不同 | 不关心可复现性 |
| `BaseDist::uniform(lo, hi, 42)` | 每次调用产生**不同**张量 | 相同 | 需要确定性序列且隔离 RNG |

**说明：**

- `generate_tensor` 的 `seed` 参数每次调用都重置全局 RNG（`torch::manual_seed`），因此同参数反复调用总是得到同一个张量。
- `BaseDist` 的 `seed` 参数创建隔离的 `torch::Generator`，其状态在调用间自动推进。同一个 `BaseDist` 实例的连续调用产生不同但确定性的张量序列，且不影响全局 RNG。
- 当 `BaseDist` 携带 Generator 时，`generate_tensor` 的 `seed` 参数无效——所有 RNG 采样均使用隔离的 Generator。
- 可通过 `test_rng_trajectory` 程序验证上述行为：运行两次并 diff 输出即可确认跨运行确定性。
