# tensor_generator

一个用于生成具有受控分布的 libtorch 张量的 C++ 库——将离散特殊值（NaN、Inf、dtype 最大/最小值等）与连续/整数基础分布（uniform/normal/randint）组合使用。

专为测试需要处理各种 PyTorch 标量类型边界情况的数值代码而设计。

## 特性

- **支持 15 种 dtype**：uint8/16/32/64、int8/16/32/64、float16/bfloat16/float32/float64、bool、float8_e4m3fn、float8_e5m2
- **离散特殊值**：Zero、One、NOne、DTypeMin、DTypeMax、NaN、Inf、NInf、Epsilon、Tiny
- **基础分布**：Uniform、Normal、RandInt（带 dtype 兼容性校验）
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
```

## 便捷接口

```cpp
#include "convenience.h"

// --- 基础分布生成器 ---
auto t1 = make_uniform_tensor({1000}, torch::kFloat32, 0.0, 1.0);
auto t2 = make_normal_tensor({1000}, torch::kFloat64, 0.0, 1.0);
auto t3 = make_randint_tensor({1000}, torch::kInt32, -100, 100);

// --- 单类特殊值（10 个命名函数） ---
auto t4 = tensor_with_nan({1000}, torch::kFloat32, 0.1);
auto t5 = tensor_with_inf({1000}, torch::kFloat32, 0.05);
auto t6 = tensor_with_zero({1000}, torch::kFloat32, 0.2);
// 还有：tensor_with_one, tensor_with_none, tensor_with_min, tensor_with_max,
//       tensor_with_ninf, tensor_with_epsilon, tensor_with_tiny

// --- 通用单类特殊值 ---
auto t7 = tensor_with_one_sv(SpecialValue::NaN, {1000}, torch::kFloat32, 0.1);

// --- 双类特殊值（45 个命名函数） ---
auto t8 = tensor_with_inf_nan({1000}, torch::kFloat32, 0.1, 0.05);
auto t9 = tensor_with_max_min({1000}, torch::kInt32, 0.2, 0.2);
// 全部 C(10,2) = 45 对组合均可用

// --- 通用双类特殊值 ---
auto t10 = tensor_with_two_sv(
    SpecialValue::NaN, 0.1, SpecialValue::Inf, 0.05,
    {1000}, torch::kFloat32);

// --- 使用 Generator 确保可复现 ---
auto gen = at::detail::createCPUGenerator();
gen.set_current_seed(42);
auto t11 = tensor_with_nan({1000}, torch::kFloat32, 0.1, gen);
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

// --- 遍历生成张量 ---
for (auto& tc : cases) {
    auto tensor = generate_tensor({1000}, torch::kFloat32, tc.spec, tc.dist);
    // 使用 tensor 测试你的数值代码...
}
```

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
BaseDist{}                          // Uniform(-1, 1)
BaseDist::uniform(lo, hi)           // 均匀分布 [lo, hi]
BaseDist::normal(mean, std)         // 正态分布 N(mean, std)
BaseDist::randint(lo, hi)           // 整数均匀分布 [lo, hi)
BaseDist(torch::Generator gen)      // 隐式转换：Uniform(-1,1) + generator
dist.with_generator(gen)            // 为任意分布绑定 generator
```

**Dtype/BaseDist 兼容性：**

| BaseDist | 浮点类型 | 整数类型 | Bool |
|----------|:---:|:---:|:---:|
| Uniform | 可用 | 可用（截断） | 可用 |
| Normal | 可用 | 可用（截断） | 可用 |
| RandInt | 报错 | 可用 | 报错 |

**SpecialValue 枚举：**

| 名称 | 描述 |
|------|------|
| `Zero` | 加法单位元 |
| `One` | 乘法单位元 |
| `NOne` | -1（无符号类型为 0xFF...FF） |
| `DTypeMin` | 类型可表示的最小值 |
| `DTypeMax` | 类型可表示的最大值 |
| `NaN` | 静默 NaN（仅浮点类型） |
| `Inf` | +无穷大（浮点类型，不含 e4m3fn） |
| `NInf` | -无穷大（浮点类型，不含 e4m3fn） |
| `Epsilon` | 机器精度（仅浮点类型） |
| `Tiny` | 最小正规化数（仅浮点类型） |

## 项目结构

```
src/
  tensor_generator.h/.cpp  — 公开 API（generate_tensor）
  tensor_spec.h            — TensorSpec、BaseDist、DiscreteEntry
  special_value.h/.cpp     — SpecialValue 枚举与 dtype 感知的值表
  dtype_traits.h           — 内部 dtype 分类辅助函数
  convenience.h            — 便捷封装接口（仅头文件）
  strategies.h/.cpp        — 策略生成器（TestCase、随机/网格/两两组合）
test/
  tensor_generator_test.cpp         — 1700+ 核心测试用例
  test_configs.h                    — 共享测试配置
  test_convenience_strategies.cpp   — 60 个便捷接口与策略测试
docs/
  design.md                — 详细设计文档
```

## 许可证

内部使用。
