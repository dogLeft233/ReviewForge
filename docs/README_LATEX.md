# ReviewForge 论文编译工作流程

本文档记录论文 `paper_cjc.tex` 从模板到成功编译的完整流程、所有修改和踩过的坑。

---

## 1. 文件清单

| 文件 | 说明 |
|------|------|
| `paper_cjc.tex` | 论文主文件 |
| `CjC.cls` | 《计算机学报》文档类（已修改） |
| `CjC_template_tex.tex` | 原始模板（未修改，供参考） |
| `ref.bib` | 参考文献（GB/T 7714 格式） |
| `picins.sty` | 过时宏包（CjC.cls 依赖，TeX Live 2024+ 已移除） |
| `figures/1_new.png` | GPT Image 2 生成的系统架构图 |
| `figures/2.png` | GPT Image 2 生成的 BFS 搜索算法图 |
| `figures/3.png` | GPT Image 2 生成的适配管线图 |
| `1_new.jpg`, `2.jpg`, `3.jpg` | 从 PNG 转换的 JPG（XeTeX 兼容性更好） |
| `figures/ui_*.png` | Playwright 截取的 Streamlit UI 截图（6张） |
| `figures/GPT_IMAGE2_PROMPTS.md` | 生成三张架构图的提示词 |

---

## 2. 编译环境

### 必需软件
- **XeLaTeX**（CJC 模板必须用 XeLaTeX，不能用 pdfLaTeX）
- **TeX Live 2017+**（推荐 2024/2025）
- **中文字体**：SimSun, SimHei, KaiTi, FangSong 或 Noto CJK / Fandol

### 本地安装命令
```bash
# Ubuntu / Debian
apt-get install texlive-xetex texlive-latex-extra texlive-bibtex-extra \
                texlive-lang-chinese fonts-noto-cjk poppler-utils

# macOS (MacTeX)
# https://tug.org/mactex/
```

---

## 3. CjC.cls 修改

### 做的唯一修改

全局替换 `\zihao` → `\cjczihao`（共 29 处）。

**原因**：原始 `CjC.cls` 用 `\newcommand\zihao` 定义了自己的字号命令。现代 `ctex` 宏包也定义了 `\zihao`，两者冲突导致编译报错：
```
LaTeX cmd Error: Command '\zihao' already defined.
```

**修复策略**：
1. CjC.cls 内部全部改用 `\cjczihao`，不与 ctex 冲突
2. 论文 preamble 中：`\usepackage{ctex}`（得到 `\heiti` `\songti` `\fangsong` `\kaishu` 等字体命令），然后用 `\let\zihao\cjczihao` 把 ctex 字号覆盖为 CjC 字号

```bash
# 修改命令（使用 sed，单引号防止转义问题）
sed -i 's/\\zihao/\\cjczihao/g' CjC.cls
```

### 其他 CjC.cls 依赖问题
- `\RequirePackage{picins}` — picins 是过时宏包，TeX Live 2024+ 已移除。从 CTAN 下载放入同目录
- `\RequirePackage{ccaption}` — 依赖 picins，但本身在 TeX Live 中可用
- `\usepackage{flushend}` — 可能触发 `stfloats` 未找到，但通常无害

---

## 4. 论文 preamble 关键设置

```latex
\documentclass[10.5pt,compsoc,UTF8]{CjC}

% 中文字体方案
\usepackage[fontset=none]{ctex}          % fontset=none 避免找 Fandol
\setCJKmainfont{Noto Serif CJK SC}[BoldFont=Noto Sans CJK SC Bold]
\setCJKsansfont{Noto Sans CJK SC}[BoldFont=Noto Sans CJK SC Bold]
% ... 设置 zhhei/zhsong/zhkai/zhfs 等 family

% 字号覆盖（关键！）
\let\zihao\cjczihao                     % 用 CjC 字号替代 ctex 字号
```

### 字体可用性问题
ctex 默认在 Linux 上找 Fandol 字体，但 Fandol 不一定安装。
- **方案A**：安装 `fonts-noto-cjk` + `texlive-lang-chinese`，在 preamble 手动指定
- **方案B**：安装 Fandol 字体包
- **Overleaf**：自动处理字体，不需要手动配置

---

## 5. 编译命令

```bash
cd docs/

# 完整编译链（4 步）
TEXINPUTS=.:$TEXINPUTS xelatex -interaction=nonstopmode paper_cjc.tex
bibtex paper_cjc
TEXINPUTS=.:$TEXINPUTS xelatex -interaction=nonstopmode paper_cjc.tex
TEXINPUTS=.:$TEXINPUTS xelatex -interaction=nonstopmode paper_cjc.tex
```

`TEXINPUTS=.:$TEXINPUTS` 确保 XeLaTeX 能在当前目录找到 `CjC.cls`、`picins.sty`、图片等文件。

### 清理命令
```bash
rm -f paper_cjc.pdf paper_cjc.aux paper_cjc.log \
      paper_cjc.bbl paper_cjc.blg paper_cjc.out paper_cjc.toc
```

---

## 6. 图片问题与修复

### 问题1：图片不显示（0 页输出）

**现象**：PDF 只有文本，所有图片不出现。

**根因**：论文正文包裹在 `\begin{multicols}{1}...\end{multicols}` 中。`multicols` 环境**不允许浮动体**（`\begin{figure}[htbp]` 的图被静默丢弃）。

**修复**：将 `[htbp]` 改为 `[H]`（`float` 宏包的非浮动模式），并在每张图前后关/开 `multicols` 让图拥有全页宽度：

```latex
\end{multicols}
\begin{figure}[H]
    \centerline{\includegraphics[width=\textwidth]{1_new.jpg}}
    \caption{...}
\end{figure}
\begin{multicols}{1}
```

### 问题2：正文文字与图表重叠

**现象**：Figure 3 的右半部分被 "5.3 引用质量评估" 等正文文字覆盖。

**根因**：图在 `multicols` 内部使用 `[H]`，列宽限制导致图文重叠。

**修复**：同问题1——每张图前后关/开 `multicols`。

### 问题3：PNG 被 XeTeX 误识别为 BMP

**现象**：编译日志显示 `File: 1_new.png Graphic file (type bmp)`，图片不嵌入 PDF。

**修复**：将 PNG 转换为 JPG。
```python
from PIL import Image
img = Image.open('1_new.png')
if img.mode == 'RGBA':
    img = img.convert('RGB')
img.save('1_new.jpg', format='JPEG', quality=95)
```

### 问题4：`Division by 0` 错误

**现象**：`! Package graphics Error: Division by 0.`

**根因**：`\includegraphics[width=0.92\textwidth]` 在 `\onecolumn` 模式下 `\textwidth` 计算为 0。

**修复**：改用绝对宽度 `width=14cm` 或在图外关闭 `multicols` 后用 `width=\textwidth`（此时 `\textwidth` 为全页宽，正常值）。

---

## 7. 常见编译错误速查

| 错误信息 | 原因 | 修复 |
|---------|------|------|
| `Command '\zihao' already defined` | CjC.cls 和 ctex 冲突 | 改名 CjC.cls 的 zihao → cjczihao |
| `File 'picins.sty' not found` | 过时宏包被移除 | 从 CTAN 下载放入目录 |
| `File 'captionhack.sty' not found` | 过时宏包 | 删除 `\usepackage{captionhack}` |
| `File 'gbt7714.sty' not found` | 未安装 bibtex-extra | `apt-get install texlive-bibtex-extra` |
| `File 'CTEX.sty' not found` | 路径大小写敏感 | 改为 `\usepackage{ctex}`（小写） |
| `Font ... not contain requested Script "CJK"` | 缺中文字体 | 安装 `fonts-noto-cjk` 或改用可用字体 |
| `Floats not allowed inside multicols` | 浮动图在 multicols 中失效 | 改用 `[H]` + 关/开 multicols |
| `Unable to load picture 'photo.jpg'` | 缺作者照片 | 删除或替换 biography 照片路径 |
| `Division by 0` (graphics) | `\textwidth=0` | 用绝对宽度或确保在 multicols 外部 |

---

## 8. Overleaf 上传

### 打包命令
```bash
python3 -c "
import zipfile, os
os.chdir('/mnt/e/Documents/ReviewForge/docs')
files = ['paper_cjc.tex', 'CjC.cls', 'ref.bib', 'picins.sty',
         '1_new.jpg', '2.jpg', '3.jpg']
with zipfile.ZipFile('../reviewforge_paper.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        zf.write(f, arcname=f)
"
```

### Overleaf 设置
- **Compiler**：XeLaTeX
- **Main document**：`paper_cjc.tex`
- 文件必须打平到项目根目录（不能有子文件夹）

### 注意事项
- `\graphicspath{{./}}` 而非 `{{figures/}}`（因为图片和 .tex 文件同级）
- `\usepackage{CTEX}` → `\usepackage{ctex}`（Overleaf 服务器是 Linux，区分大小写）
- Overleaf 已内置中文字体和 Fandol，不需要 `fontset=none` 的手动字体配置

---

## 9. Git 工作流

论文文件在 `docs/` 目录下，但 `docs/` 在 `.gitignore` 中。提交时需 `-f` 强制添加：

```bash
git add -f docs/paper_cjc.tex docs/CjC.cls docs/ref.bib docs/picins.sty \
           docs/figures/ docs/1_new.jpg docs/2.jpg docs/3.jpg

git commit -m "feat: paper update description"
```

编译产物（`paper_cjc.pdf`, `*.aux`, `*.log` 等）不要提交。

---

## 10. 论文当前状态

- **页数**：10 页
- **大小**：~2.5 MB（含 3 张嵌入图片）
- **章节**：引言 → 相关工作 → 系统架构 → 关键模块设计 → 实验评估 → 结论
- **参考文献**：20 条（GB/T 7714 格式）
- **图片**：3 张架构图（GPT Image 2 生成）
- **待完成**：作者信息、传记照片、基金致谢、`Background` 段落位置调整
