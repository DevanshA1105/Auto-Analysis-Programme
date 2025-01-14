# Auto-Analysis Programme

This repository is for Project 2 of Tools in Data Science, focusing on automated data analysis using large language models (LLMs).

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
The Auto-Analysis Programme is designed to automate various data analysis tasks using LLMs. This project aims to provide an efficient and robust solution for analyzing data with minimal manual intervention.

## Features
- Automated data preprocessing and cleaning
- Data visualization and exploratory data analysis
- Statistical analysis and hypothesis testing
- Machine learning model training and evaluation
- Report generation and result summarization

## Installation
Install [uv](https://docs.astral.sh/uv/guides/scripts) for running the project code.
```bash
pip install uv
```

## Usage
- Set an [OpenAI API Proxy](https://platform.openai.com/docs/api-reference/introduction) Key as an environment variable using the following command. (Replace ABCDEFGH with your actual API key)
```bash
export 'AIPROXY_TOKEN' = ABCDEFGH
```
- Run the following command to start the auto analysis.
```bash
uv run path_to_your_file
```
