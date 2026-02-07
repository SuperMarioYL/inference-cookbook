# vLLM Cookbook

> 面向深度学习初学者的 vLLM 技术文档，从基础概念到核心算法的完整解析

**在线阅读**: [https://inference.lei6393.com](https://inference.lei6393.com)

## 简介

本文档系列旨在帮助深度学习初学者深入理解 vLLM —— 一个高性能的大语言模型（LLM）推理和服务框架。我们将从最基础的概念出发，逐步深入到核心算法和代码实现，让你不仅知其然，更知其所以然。

## 你将学到

- 大语言模型推理面临的核心挑战
- Transformer 架构和注意力机制的工作原理
- vLLM 的核心创新：PagedAttention 和连续批处理
- 从入口到输出的完整代码执行链路
- 如何调试和分析 vLLM 代码

## 目录结构

```
content/docs/
├── 00-overview/          # 扬帆起航 - 文档概览
├── 01-introduction/      # 入门篇
├── 02-dl-fundamentals/   # 深度学习基础
├── 03-core-modules/      # 核心模块详解
├── 04-code-walkthrough/  # 代码链路分析
├── 05-advanced-topics/   # 进阶主题
└── appendix/             # 附录
```

## 本地开发

### 前置要求

- [Node.js](https://nodejs.org/) v20+
- [Hugo](https://gohugo.io/installation/) Extended 版本 v0.147.0+

### 启动本地服务器

```bash
# 克隆仓库
git clone https://github.com/SuperMarioYL/vllm-book.git
cd vllm-book

# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 访问 http://localhost:1313
```

### 构建

```bash
npm run build
```

构建产物在 `public/` 目录下。

## 技术栈

- [Hugo](https://gohugo.io/) - 静态网站生成器
- [Doks](https://getdoks.org/) - 文档主题
- [GitHub Pages](https://pages.github.com/) - 托管平台
- [GitHub Actions](https://github.com/features/actions) - CI/CD

## 许可

MIT License
