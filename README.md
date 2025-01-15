# Auto-Analysis Programme

This repository is for Project 2 of Tools in Data Science, focusing on automated data analysis using large language models (LLMs).

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Samples](#samples)
- [Installations](#installations)
- [Usage](#usage)

## Introduction
The Auto-Analysis Programme is designed to automate various data analysis tasks using LLMs. This project aims to provide an efficient and robust solution for analyzing data with minimal manual intervention.

## Features
- Automated data preprocessing and cleaning
- Data visualization and exploratory data analysis
- Statistical analysis and hypothesis testing
- LLM API calling and tooling

## Samples
Some sample results from the analysis have been shown in the folders. Each folder includes some graphs and a README.md file which summarizes the analysis process.
The datasets used for generating these samples were provided by [Mr. Anand S](https://sg.linkedin.com/in/sanand0) , and can be found [here](https://github.com/sanand0/tools-in-data-science-public/blob/tds-2024-t3/project-2-automated-analysis.md) along with further project details.

## Installations
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
uv run autolysis.py path_to_your_file
```
