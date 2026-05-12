# 超声无损检测信号快速分析软件项目需求说明

## 1. 项目定位

本项目旨在开发一款面向科研场景的信号快速查看、处理、分析与导出软件，主要服务于超声无损检测、声发射、振动信号、传感器阵列信号等实验数据的快速预处理与特征分析。

软件应支持从 CSV、TXT、MAT、NPY 等常见数据文件中读取单通道或多通道信号，实现信号可视化、窄带提取、小波变换、特征计算、多通道对比、参数调节、结果导出，并预留与 MATLAB、COMSOL、AI 模型接口集成的扩展能力。

---

## 2. 目标用户

主要用户包括：

1. 超声无损检测方向研究生、博士生、科研人员。
2. 需要快速查看实验信号的工程师。
3. 需要批量处理传感器数据的研究人员。
4. 需要与 MATLAB、COMSOL 仿真结果对比的用户。

---

## 3. 核心使用场景

### 3.1 快速查看实验信号

用户导入一个 CSV 文件，软件自动识别时间列、信号列、采样频率，并绘制时域波形。用户可以缩放、拖动、选区、查看峰值、到达时间、信号长度等基本信息。

### 3.2 窄带提取

用户选择某个中心频率和带宽，例如 `500 kHz ± 50 kHz`，软件对原始信号进行带通滤波，显示滤波前后的对比结果，并允许导出处理后的信号。

### 3.3 小波变换分析

用户选择小波类型、尺度范围、频率范围、采样率等参数，软件生成时频图，并允许用户调节参数实时观察结果变化。

### 3.4 多通道信号对比

用户导入最多 8 个通道的信号，软件在同一界面显示多通道波形，支持统一时间轴、幅值归一化、延迟估计、互相关、到达时间比较等功能。

### 3.5 仿真结果对比

用户从 COMSOL 或 MATLAB 导入仿真信号，与实验信号进行对比，包括时域曲线、频谱、小波图、峰值时间、主频、能量等。

### 3.6 AI 辅助识别

用户导入未知数据文件后，AI 模块辅助识别文件结构，例如时间列、信号列、采样频率、单位、通道数、是否包含表头等，并给出导入建议。

---

# 4. 功能需求

## 4.1 数据输入模块

### 4.1.1 支持文件格式

第一阶段建议支持：

| 格式 | 说明 |
|---|---|
| `.csv` | 最优先支持，实验数据常用格式 |
| `.txt` | 支持空格、Tab、逗号分隔 |
| `.xlsx` | 可选，便于处理实验记录 |
| `.mat` | MATLAB 数据文件 |
| `.npy` / `.npz` | Python NumPy 数据 |
| `.json` | 可选，用于结构化参数或元数据 |

第二阶段可支持：

| 格式 | 说明 |
|---|---|
| `.h5` / `.hdf5` | 大规模实验数据 |
| `.tdms` | LabVIEW 常见数据格式 |
| COMSOL 导出文件 | 通常为 `.txt`、`.csv`、`.mph` 间接导出数据 |

### 4.1.2 文件导入要求

软件应支持：

1. 单文件导入。
2. 多文件批量导入。
3. 拖拽导入。
4. 自动识别分隔符。
5. 自动识别是否存在表头。
6. 自动识别时间列。
7. 自动识别信号列。
8. 支持用户手动指定采样频率。
9. 支持用户手动指定单位。
10. 支持用户选择导入的通道。

### 4.1.3 数据结构要求

软件内部建议统一转换为如下结构：

```python
@dataclass
class SignalData:
    name: str
    time: np.ndarray
    values: np.ndarray
    sample_rate: float
    channel_names: list[str]
    unit_time: str = "s"
    unit_amplitude: str = "V"
    metadata: dict = field(default_factory=dict)
```

对于多通道数据：

```python
@dataclass
class MultiChannelSignal:
    name: str
    time: np.ndarray
    channels: dict[str, np.ndarray]
    sample_rate: float
    metadata: dict = field(default_factory=dict)
```

---

## 4.2 信号显示模块

### 4.2.1 时域显示

应支持：

1. 单通道波形显示。
2. 多通道波形显示。
3. 最多 8 通道同时显示。
4. 鼠标缩放。
5. 鼠标拖动。
6. 框选局部区域。
7. 显示当前鼠标位置的时间与幅值。
8. 支持恢复全局视图。
9. 支持信号归一化显示。
10. 支持叠加显示或分图显示。

### 4.2.2 频域显示

应支持：

1. FFT 频谱。
2. 幅值谱。
3. 功率谱。
4. 对数坐标显示。
5. 频率范围选择。
6. 主频自动识别。
7. 频谱峰值标注。
8. 多通道频谱对比。

### 4.2.3 时频显示

应支持：

1. 连续小波变换 CWT。
2. 短时傅里叶变换 STFT，可选。
3. 时频图显示。
4. 频率范围调节。
5. 颜色条显示。
6. 幅值归一化。
7. 对数频率轴，可选。
8. 图像导出。

---

## 4.3 窄带提取模块

### 4.3.1 功能目标

用于从宽带超声信号中提取指定频率附近的成分。

### 4.3.2 参数

用户可调节：

| 参数 | 说明 |
|---|---|
| 中心频率 | 例如 500 kHz |
| 带宽 | 例如 100 kHz |
| 下限频率 | 可由中心频率和带宽自动计算 |
| 上限频率 | 可由中心频率和带宽自动计算 |
| 滤波器类型 | Butterworth、FIR、Chebyshev 可选 |
| 滤波器阶数 | 默认 4 或 6 |
| 是否零相位滤波 | 默认使用 `filtfilt` |
| 是否归一化 | 可选 |

### 4.3.3 输出

应输出：

1. 窄带信号。
2. 原始信号与窄带信号对比图。
3. 窄带信号频谱。
4. 滤波器频率响应，可选。
5. 可导出 CSV。

---

## 4.4 特征分析模块

### 4.4.1 时域特征

应支持计算：

| 特征 | 说明 |
|---|---|
| 最大值 | 信号最大幅值 |
| 最小值 | 信号最小幅值 |
| 峰峰值 | max - min |
| RMS | 均方根 |
| 平均值 | mean |
| 标准差 | std |
| 能量 | sum(x²) |
| 信号持续时间 | 超过阈值的时间范围 |
| 到达时间 TOF | 首次超过阈值的时间 |
| 峰值时间 | 最大幅值对应时间 |
| 包络峰值 | Hilbert 包络最大值 |
| 包络峰值时间 | 包络最大值对应时间 |

### 4.4.2 频域特征

应支持计算：

| 特征 | 说明 |
|---|---|
| 主频 | 频谱最大峰值对应频率 |
| 频谱质心 | spectral centroid |
| 带宽 | 根据能量分布计算 |
| -3 dB 带宽 | 可选 |
| 高频能量比例 | 指定频带内能量比例 |
| 低频能量比例 | 指定频带内能量比例 |
| 频谱峰值数量 | 可选 |

### 4.4.3 多通道特征

应支持：

1. 通道间峰值时间差。
2. 通道间到达时间差。
3. 通道间互相关。
4. 最大相关系数。
5. 延迟估计。
6. 能量比。
7. 主频差异。
8. 波形相似度。

---

## 4.5 小波变换模块

### 4.5.1 目标

用于分析超声信号的时频特征，尤其适合短脉冲、非平稳信号、缺陷回波信号等。

### 4.5.2 支持方法

第一阶段建议支持：

1. CWT 连续小波变换。
2. Morlet 小波。
3. Mexican hat 小波。
4. Complex Morlet 小波。

第二阶段可支持：

1. DWT 离散小波分解。
2. 小波包分解。
3. 多尺度能量分析。
4. 小波去噪。

### 4.5.3 可调参数

| 参数 | 说明 |
|---|---|
| 小波类型 | Morlet、Mexican hat、Complex Morlet |
| 采样频率 | 自动识别或手动输入 |
| 最小频率 | 用户设定 |
| 最大频率 | 用户设定 |
| 频率点数 | 控制时频图分辨率 |
| 尺度范围 | 可自动由频率范围换算 |
| 颜色映射 | viridis、jet、gray 等 |
| 是否归一化 | 可选 |
| 是否显示对数频率轴 | 可选 |

### 4.5.4 输出

应支持：

1. 小波时频图。
2. 每个时间点的 dominant frequency。
3. 小波能量分布。
4. 指定频带的小波能量曲线。
5. 图像导出。
6. 小波结果数据导出。

---

## 4.6 AI 辅助识别模块

### 4.6.1 目标

AI 模块不应作为第一阶段核心依赖，而应设计成可选扩展模块。软件即使没有 AI 接口，也应能正常完成信号处理。

### 4.6.2 功能

AI 接入后可以辅助完成：

1. 识别 CSV 文件结构。
2. 判断是否有表头。
3. 判断时间列、信号列、通道列。
4. 推测采样频率。
5. 推测单位，例如 s、ms、μs、V、mV。
6. 根据文件名或列名识别实验工况。
7. 自动生成数据摘要。
8. 自动推荐滤波参数。
9. 自动识别异常信号。
10. 自动生成分析报告初稿。

### 4.6.3 AI 接口设计

建议抽象为统一接口：

```python
class AIAnalyzer:
    def analyze_file_structure(self, file_path: str) -> dict:
        pass

    def infer_sampling_rate(self, data_preview: dict) -> float:
        pass

    def suggest_processing_params(self, signal_info: dict) -> dict:
        pass

    def summarize_signal_features(self, features: dict) -> str:
        pass
```

AI 返回格式建议为 JSON：

```json
{
  "has_header": true,
  "time_column": "Time",
  "signal_columns": ["CH1", "CH2"],
  "sample_rate": 10000000,
  "time_unit": "s",
  "amplitude_unit": "V",
  "confidence": 0.92,
  "notes": "Detected uniform time interval of 1e-7 s."
}
```

---

## 4.7 MATLAB / COMSOL 接口模块

### 4.7.1 MATLAB 接口

建议支持三种方式。

#### 方式一：读取 `.mat` 文件

这是最容易实现、最稳定的方式。

功能要求：

1. 读取 MATLAB `.mat` 文件。
2. 允许用户选择变量。
3. 自动识别一维数组、二维数组。
4. 自动转换为内部信号格式。
5. 支持导出为 `.mat`，可选。

#### 方式二：调用 MATLAB Engine

适合高级用户。

功能要求：

1. Python 调用 MATLAB Engine。
2. 将当前信号传入 MATLAB。
3. 运行用户指定 `.m` 脚本。
4. 读取 MATLAB 处理结果。
5. 将结果显示在软件中。

示例接口：

```python
class MatlabBridge:
    def load_mat_file(self, path: str) -> dict:
        pass

    def run_script(self, script_path: str, variables: dict) -> dict:
        pass
```

### 4.7.2 COMSOL 接口

COMSOL 的 `.mph` 文件直接解析较复杂，第一阶段建议不直接读取 `.mph`，而支持 COMSOL 导出的文本或 CSV 数据。

第一阶段支持：

1. 读取 COMSOL 导出的 `.txt`、`.csv`。
2. 自动跳过 COMSOL 文件头部说明。
3. 识别时间列、位移列、速度列、应力列等。
4. 将仿真信号与实验信号对比。

第二阶段支持：

1. LiveLink for MATLAB。
2. COMSOL Java API。
3. 通过脚本自动导出指定探针点信号。
4. 与实验数据自动对齐。

---

## 4.8 多通道对比模块

### 4.8.1 通道数量

最多支持 8 通道同时显示和分析。

### 4.8.2 显示方式

应支持：

1. 多通道叠加显示。
2. 多通道分图显示。
3. 统一时间轴。
4. 独立幅值轴。
5. 全局归一化。
6. 每通道独立归一化。
7. 通道颜色区分。
8. 通道隐藏 / 显示。

### 4.8.3 分析功能

应支持：

1. 通道间互相关。
2. 延迟估计。
3. 峰值到达时间差。
4. 主频对比。
5. 能量对比。
6. 特征表格汇总。
7. 选择一个通道作为参考通道。

---

## 4.9 数据输出模块

### 4.9.1 CSV 输出

应支持导出：

1. 原始信号。
2. 滤波后信号。
3. 窄带信号。
4. 包络信号。
5. 小波能量曲线。
6. 多通道对齐结果。
7. 特征分析表格。

### 4.9.2 图像输出

应支持：

1. 时域图导出。
2. 频谱图导出。
3. 小波时频图导出。
4. 多通道对比图导出。

格式建议：

```text
.png
.svg
.pdf
```

### 4.9.3 报告输出，可选

第二阶段可增加：

1. 自动生成 Markdown 报告。
2. 自动生成 HTML 报告。
3. 自动生成 PDF 报告。
4. 包含参数、图像、特征表格、处理流程。

---

# 5. 软件架构建议

## 5.1 推荐技术路线

建议采用 Python 作为主要开发语言。

桌面端推荐：

```text
Python + PySide6 / PyQt6 + pyqtgraph + scipy + numpy + pandas + pywavelets
```

理由：

1. Python 科研生态成熟。
2. scipy、numpy、pywavelets 适合信号处理。
3. PySide6 / PyQt6 适合做桌面软件。
4. pyqtgraph 适合快速交互绘图，性能比 matplotlib 更适合实时查看。
5. 后续容易接入 AI、MATLAB Engine、COMSOL 脚本。

## 5.2 替代技术路线

### 方案 A：Python 桌面软件

```text
PySide6 + pyqtgraph + scipy + pandas + pywavelets
```

优点：

1. 开发快。
2. 适合科研人员。
3. 便于后续扩展。
4. 容易打包成 exe。

缺点：

1. UI 美观度需要额外打磨。
2. 大数据性能需要优化。

### 方案 B：Web 应用

```text
FastAPI + React + Plotly
```

优点：

1. UI 更现代。
2. 跨平台。
3. 适合团队共享。

缺点：

1. 文件本地读写略复杂。
2. 部署成本高于桌面软件。
3. 与本地 MATLAB / COMSOL 集成更麻烦。

### 方案 C：MATLAB App Designer

优点：

1. 信号处理工具箱丰富。
2. 对 MATLAB 用户友好。
3. 与现有 MATLAB 脚本兼容。

缺点：

1. 可分发性较差。
2. AI 接口和现代 UI 扩展不如 Python 灵活。
3. 依赖 MATLAB 授权。

## 5.3 推荐最终方案

建议第一版使用：

```text
Python + PySide6 + pyqtgraph + numpy + scipy + pandas + pywavelets
```

后续再根据需要扩展：

```text
AI 接口：OpenAI API / 本地大模型 / Ollama
MATLAB 接口：matlab.engine
COMSOL 接口：COMSOL 导出 CSV / LiveLink for MATLAB
报告输出：Markdown / HTML / PDF
```

---

# 6. 推荐项目目录结构

```text
signal_analyzer/
│
├── main.py
├── requirements.txt
├── README.md
├── pyproject.toml
│
├── app/
│   ├── __init__.py
│   ├── main_window.py
│   ├── widgets/
│   │   ├── signal_plot_widget.py
│   │   ├── spectrum_plot_widget.py
│   │   ├── wavelet_plot_widget.py
│   │   ├── channel_selector.py
│   │   ├── parameter_panel.py
│   │   └── feature_table.py
│   │
│   ├── dialogs/
│   │   ├── import_dialog.py
│   │   ├── export_dialog.py
│   │   ├── filter_dialog.py
│   │   └── ai_dialog.py
│
├── core/
│   ├── __init__.py
│   ├── signal_data.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── filtering.py
│   ├── feature_extraction.py
│   ├── wavelet_analysis.py
│   ├── spectrum_analysis.py
│   ├── multichannel_analysis.py
│   └── export.py
│
├── bridges/
│   ├── __init__.py
│   ├── matlab_bridge.py
│   ├── comsol_bridge.py
│   └── ai_bridge.py
│
├── config/
│   ├── default_config.json
│   └── user_config.json
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_filtering.py
│   ├── test_features.py
│   ├── test_wavelet.py
│   └── test_export.py
│
└── examples/
    ├── sample_single_channel.csv
    ├── sample_multi_channel.csv
    └── sample_comsol_export.txt
```

---

# 7. 核心模块开发说明

## 7.1 数据加载模块 `data_loader.py`

职责：

1. 读取 CSV、TXT、MAT、NPY 文件。
2. 自动判断分隔符。
3. 自动判断表头。
4. 自动识别时间列。
5. 自动识别通道列。
6. 返回统一的 `SignalData` 或 `MultiChannelSignal` 对象。

核心函数建议：

```python
def load_signal_file(path: str) -> SignalData:
    pass

def detect_delimiter(path: str) -> str:
    pass

def detect_header(path: str) -> bool:
    pass

def infer_time_column(df) -> str:
    pass

def infer_sample_rate(time_array) -> float:
    pass

def select_signal_columns(df) -> list[str]:
    pass
```

## 7.2 滤波模块 `filtering.py`

职责：

1. 带通滤波。
2. 低通滤波。
3. 高通滤波。
4. 窄带提取。
5. 滤波器频响计算。

核心函数建议：

```python
def bandpass_filter(
    signal,
    sample_rate,
    lowcut,
    highcut,
    order=4,
    zero_phase=True
):
    pass

def narrowband_extract(
    signal,
    sample_rate,
    center_freq,
    bandwidth,
    order=4
):
    pass

def compute_filter_response(
    sample_rate,
    lowcut,
    highcut,
    order=4
):
    pass
```

## 7.3 特征分析模块 `feature_extraction.py`

职责：

1. 时域特征提取。
2. 频域特征提取。
3. 包络特征提取。
4. 到达时间计算。
5. 汇总为表格。

核心函数建议：

```python
def compute_time_features(time, signal, threshold=None) -> dict:
    pass

def compute_frequency_features(signal, sample_rate) -> dict:
    pass

def compute_envelope_features(time, signal) -> dict:
    pass

def estimate_arrival_time(time, signal, method="threshold", threshold_ratio=0.1) -> float:
    pass

def summarize_features(time, signal, sample_rate) -> dict:
    pass
```

## 7.4 小波分析模块 `wavelet_analysis.py`

职责：

1. 连续小波变换。
2. 小波时频图数据生成。
3. 频率尺度转换。
4. 小波能量计算。

核心函数建议：

```python
def compute_cwt(
    signal,
    sample_rate,
    wavelet="cmor1.5-1.0",
    f_min=None,
    f_max=None,
    num_freqs=200
):
    pass

def compute_wavelet_energy(cwt_matrix) -> dict:
    pass

def extract_band_energy(cwt_matrix, frequencies, f_low, f_high):
    pass
```

## 7.5 多通道分析模块 `multichannel_analysis.py`

职责：

1. 多通道对齐。
2. 通道间互相关。
3. 延迟估计。
4. 多通道特征表。

核心函数建议：

```python
def estimate_delay_by_correlation(
    ref_signal,
    target_signal,
    sample_rate
) -> float:
    pass

def compute_channel_features(
    time,
    channels,
    sample_rate
) -> dict:
    pass

def align_channels_by_delay(
    channels,
    delays,
    sample_rate
) -> dict:
    pass
```

## 7.6 导出模块 `export.py`

职责：

1. 导出 CSV。
2. 导出特征表。
3. 导出图像。
4. 导出处理记录。

核心函数建议：

```python
def export_signal_to_csv(path, time, signals: dict):
    pass

def export_features_to_csv(path, features: dict):
    pass

def export_processing_history(path, history: list):
    pass
```

---

# 8. UI 设计建议

## 8.1 主界面布局

建议采用如下布局：

```text
┌─────────────────────────────────────────────┐
│ 菜单栏：文件  处理  分析  导出  设置  帮助     │
├───────────────┬─────────────────────────────┤
│ 文件/通道列表  │ 主绘图区域                    │
│               │ ┌─────────────────────────┐ │
│ CH1           │ │ 时域图                    │ │
│ CH2           │ └─────────────────────────┘ │
│ CH3           │ ┌─────────────────────────┐ │
│               │ │ 频谱图 / 小波图           │ │
│               │ └─────────────────────────┘ │
├───────────────┼─────────────────────────────┤
│ 参数面板       │ 特征结果表格                  │
└───────────────┴─────────────────────────────┘
```

## 8.2 页面 / 标签页设计

建议设计以下标签页：

1. **时域查看**
2. **频谱分析**
3. **窄带提取**
4. **小波分析**
5. **多通道对比**
6. **导出结果**
7. **AI 辅助，可选**
8. **仿真接口，可选**

## 8.3 菜单设计

```text
文件
 ├── 打开文件
 ├── 打开文件夹
 ├── 最近打开
 ├── 保存处理结果
 └── 退出

处理
 ├── 去直流
 ├── 归一化
 ├── 带通滤波
 ├── 窄带提取
 └── 包络提取

分析
 ├── 时域特征
 ├── 频域特征
 ├── 小波变换
 ├── 多通道互相关
 └── 到达时间估计

导出
 ├── 导出 CSV
 ├── 导出图像
 ├── 导出特征表
 └── 导出报告

接口
 ├── 导入 MATLAB 数据
 ├── 导入 COMSOL 数据
 └── AI 辅助识别

设置
 ├── 默认采样率
 ├── 默认单位
 ├── 默认滤波参数
 └── AI API 设置
```

---

# 9. 处理流程设计

## 9.1 单通道典型流程

```text
导入文件
  ↓
自动识别采样频率 / 时间列
  ↓
显示时域信号
  ↓
选择处理方式
  ↓
窄带提取 / 小波变换 / 特征分析
  ↓
查看结果
  ↓
导出 CSV / 图像 / 特征表
```

## 9.2 多通道典型流程

```text
导入多通道文件
  ↓
选择最多 8 个通道
  ↓
统一时间轴显示
  ↓
选择参考通道
  ↓
计算互相关 / 延迟 / 峰值时间差
  ↓
生成多通道特征表
  ↓
导出结果
```

## 9.3 仿真对比流程

```text
导入实验信号
  ↓
导入 COMSOL / MATLAB 仿真信号
  ↓
统一采样率 / 时间范围
  ↓
归一化或幅值标定
  ↓
时域对比
  ↓
频谱对比
  ↓
小波对比
  ↓
输出对比结果
```

---

# 10. 参数管理需求

软件应保存常用参数，例如：

```json
{
  "default_sample_rate": 10000000,
  "default_time_unit": "s",
  "default_amplitude_unit": "V",
  "default_filter": {
    "type": "butterworth",
    "order": 4,
    "zero_phase": true
  },
  "default_wavelet": {
    "type": "cmor1.5-1.0",
    "f_min": 100000,
    "f_max": 2000000,
    "num_freqs": 200
  },
  "max_channels": 8
}
```

---

# 11. 处理历史记录

每次处理应记录参数，便于复现。

示例：

```json
[
  {
    "step": "load_file",
    "file": "sample.csv",
    "sample_rate": 10000000
  },
  {
    "step": "narrowband_extract",
    "center_freq": 500000,
    "bandwidth": 100000,
    "filter_type": "butterworth",
    "order": 4
  },
  {
    "step": "wavelet_transform",
    "wavelet": "cmor1.5-1.0",
    "f_min": 100000,
    "f_max": 2000000
  }
]
```

---

# 12. 非功能需求

## 12.1 性能需求

1. 能流畅显示 100 万点以内的单通道信号。
2. 对超过 100 万点的数据，应支持降采样显示。
3. 多通道最多 8 通道。
4. 普通 FFT、滤波、特征计算应在数秒内完成。
5. 小波变换可允许较长计算时间，但应有进度提示。
6. UI 不应在计算过程中卡死，应使用后台线程。

## 12.2 稳定性需求

1. 文件读取失败时应给出明确错误。
2. 采样率无法识别时，应提示用户手动输入。
3. 滤波频率超过 Nyquist 频率时，应禁止执行并提示。
4. 小波参数不合理时，应提示。
5. 导出失败时应提示路径或权限问题。
6. AI 接口失败时，不应影响基础功能。

## 12.3 可扩展性需求

1. 信号处理算法与 UI 分离。
2. 数据加载器可扩展新格式。
3. AI 接口可替换。
4. MATLAB / COMSOL 接口可选安装。
5. 处理流程可记录、复现。
6. 后续可增加插件系统。

---

# 13. 第一版 MVP 范围

建议不要一开始做太大。第一版重点是“能用、稳定、可扩展”。

## 13.1 MVP 必须实现

1. CSV / TXT 导入。
2. 单通道和多通道显示。
3. 自动识别时间列和采样率。
4. 用户手动指定采样率。
5. 时域图显示。
6. FFT 频谱显示。
7. 窄带带通滤波。
8. 基础特征计算。
9. CWT 小波变换。
10. 最多 8 通道显示。
11. 导出处理后 CSV。
12. 导出特征 CSV。

## 13.2 MVP 暂缓实现

这些功能可以第二阶段做：

1. AI 自动识别。
2. MATLAB Engine 实时调用。
3. COMSOL API 直接连接。
4. PDF 报告生成。
5. 批处理。
6. 高级小波包分析。
7. 插件系统。
8. 自动缺陷识别模型。

---

# 14. 开发阶段规划

## 阶段 1：基础框架

目标：能打开软件、导入数据、显示波形。

任务：

1. 建立项目结构。
2. 实现 `SignalData` 数据类。
3. 实现 CSV / TXT 读取。
4. 实现主窗口。
5. 实现时域图显示。
6. 实现通道选择面板。

验收标准：

1. 可以打开 CSV。
2. 可以选择通道。
3. 可以显示波形。
4. 可以查看采样率和数据点数。

## 阶段 2：基础信号处理

目标：实现常用信号处理功能。

任务：

1. 实现 FFT。
2. 实现频谱图。
3. 实现带通滤波。
4. 实现窄带提取。
5. 实现 Hilbert 包络。
6. 实现特征计算。

验收标准：

1. 可以显示频谱。
2. 可以输入中心频率和带宽。
3. 可以生成窄带信号。
4. 可以计算峰值、RMS、能量、主频等特征。

## 阶段 3：小波分析

目标：实现 CWT 小波时频分析。

任务：

1. 实现 CWT 算法模块。
2. 实现小波参数面板。
3. 实现时频图显示。
4. 实现频率范围调节。
5. 实现小波结果导出。

验收标准：

1. 可以选择小波类型。
2. 可以设置频率范围。
3. 可以生成小波时频图。
4. 可以导出图像或小波能量结果。

## 阶段 4：多通道对比

目标：支持最多 8 通道对比。

任务：

1. 实现多通道导入。
2. 实现多通道显示。
3. 实现通道隐藏 / 显示。
4. 实现通道归一化。
5. 实现互相关延迟估计。
6. 实现多通道特征表。

验收标准：

1. 可以同时显示 8 通道。
2. 可以计算通道间时间延迟。
3. 可以导出多通道特征表。

## 阶段 5：接口与扩展

目标：增加 AI、MATLAB、COMSOL 扩展能力。

任务：

1. 实现 `.mat` 文件读取。
2. 实现 COMSOL 导出 CSV / TXT 兼容读取。
3. 设计 AI 接口。
4. 实现 AI 文件结构识别。
5. 增加配置界面。

验收标准：

1. 可以读取 MATLAB `.mat` 数据。
2. 可以读取常见 COMSOL 导出数据。
3. AI 可返回文件结构建议。
4. 即使没有 AI key，软件仍可正常使用。

---

# 15. 推荐依赖

```txt
numpy
scipy
pandas
matplotlib
pyqtgraph
PySide6
PyWavelets
openpyxl
scikit-learn
h5py
```

可选依赖：

```txt
matlabengine
openai
ollama
hdf5storage
nptdms
```

---

# 16. 给 Codex / Agent 的开发提示词

```text
请帮我开发一个 Python 桌面软件，用于超声无损检测信号的快速查看、处理和分析。

技术栈：
- Python
- PySide6
- pyqtgraph
- numpy
- scipy
- pandas
- PyWavelets

核心需求：
1. 支持 CSV / TXT 导入。
2. 自动识别时间列、信号列、采样频率。
3. 支持用户手动指定采样频率。
4. 支持最多 8 通道信号同时显示。
5. 支持时域波形显示、缩放、拖动。
6. 支持 FFT 频谱分析。
7. 支持窄带提取，用户可设置中心频率、带宽、滤波器阶数。
8. 支持基础特征计算，包括峰值、峰峰值、RMS、能量、主频、到达时间、包络峰值。
9. 支持连续小波变换 CWT，用户可调节小波类型、频率范围、频率点数。
10. 支持导出处理后信号和特征表为 CSV。
11. 代码结构应清晰，UI 与算法分离。
12. 预留 AI 接口、MATLAB 接口、COMSOL 导入接口。

请先创建项目结构，然后实现 MVP：
- main.py
- app/main_window.py
- app/widgets/signal_plot_widget.py
- core/signal_data.py
- core/data_loader.py
- core/filtering.py
- core/spectrum_analysis.py
- core/feature_extraction.py
- core/wavelet_analysis.py
- core/export.py

要求：
- 每个模块写清楚 docstring。
- 所有核心算法写单元测试。
- 对异常文件、采样率缺失、滤波参数错误进行处理。
- UI 中应包含导入文件、通道选择、参数设置、时域图、频谱图、特征表、导出按钮。
```

---

# 17. 建议优先实现的核心算法函数

第一批可以让 Agent 直接写这些函数：

```python
def infer_sample_rate_from_time(time):
    """
    根据时间数组推断采样频率。
    要求检查时间间隔是否近似均匀。
    """
    pass
```

```python
def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    """
    使用 Butterworth 带通滤波器进行零相位滤波。
    """
    pass
```

```python
def compute_fft(signal, fs):
    """
    计算单边幅值谱。
    """
    pass
```

```python
def compute_basic_features(time, signal, fs):
    """
    计算峰值、峰峰值、RMS、能量、主频、峰值时间等。
    """
    pass
```

```python
def compute_cwt(signal, fs, f_min, f_max, num_freqs, wavelet):
    """
    根据频率范围计算连续小波变换。
    """
    pass
```

```python
def estimate_delay_by_xcorr(sig_ref, sig_target, fs):
    """
    通过互相关估计两个通道之间的时间延迟。
    """
    pass
```

---

# 18. 推荐验收数据格式

## 18.1 单通道 CSV

```csv
time,signal
0.0000000,0.001
0.0000001,0.003
0.0000002,0.010
0.0000003,0.025
```

## 18.2 多通道 CSV

```csv
time,CH1,CH2,CH3,CH4
0.0000000,0.001,0.002,0.001,0.000
0.0000001,0.003,0.004,0.002,0.001
0.0000002,0.010,0.012,0.008,0.006
```

## 18.3 无时间列 CSV

```csv
CH1,CH2,CH3
0.001,0.002,0.001
0.003,0.004,0.002
0.010,0.012,0.008
```

这种情况下软件应提示用户输入采样频率，然后自动生成时间轴。

---

# 19. 关键设计建议

## 19.1 不要让 AI 成为核心依赖

AI 很适合辅助识别文件结构、总结结果、推荐参数，但不要让软件的基础读取、滤波、分析依赖 AI。核心算法应完全本地可运行。

## 19.2 优先保证数据处理可复现

每次处理都要记录：

1. 输入文件。
2. 采样率。
3. 滤波参数。
4. 小波参数。
5. 通道选择。
6. 导出路径。
7. 软件版本。

这对科研很重要。

## 19.3 UI 先简单，算法先稳定

第一版不要追求过度美观。建议先做出稳定的工具型界面：

1. 左侧文件和通道。
2. 中间图像。
3. 右侧参数。
4. 下方特征表。

## 19.4 大数据绘图要降采样

超声信号采样率高，数据点可能很多。绘图时不要直接画全部点。可以：

1. 原始数据用于计算。
2. 降采样数据用于显示。
3. 用户放大局部时再显示更高精度数据。

---

# 20. 最小可交付版本定义

一个合格的第一版应做到：

1. 用户可以导入 CSV。
2. 用户可以看到时域波形。
3. 用户可以选择通道。
4. 用户可以输入采样率。
5. 用户可以做 FFT。
6. 用户可以做窄带提取。
7. 用户可以计算基础特征。
8. 用户可以做 CWT 小波变换。
9. 用户可以导出处理结果为 CSV。
10. 软件不会因为常见错误输入直接崩溃。

---

# 21. 推荐开发顺序

建议严格按下面顺序开发：

```text
1. 数据结构 SignalData
2. CSV/TXT 数据读取
3. 时域绘图
4. 通道选择
5. FFT 频谱
6. 带通滤波 / 窄带提取
7. 特征计算
8. CSV 导出
9. 小波变换
10. 多通道对比
11. MATLAB / COMSOL 文件兼容
12. AI 辅助识别
13. 报告生成
14. 打包发布
```

---

# 22. 项目一句话总结

这个软件的核心不是“复杂平台”，而是一个 **面向超声无损检测实验数据的快速信号查看、窄带提取、时频分析、多通道对比和结果导出工具**。第一版应优先做到稳定读取、快速显示、参数可调、结果可复现、CSV 可导出；AI、COMSOL、MATLAB 接口作为后续扩展模块逐步加入。
