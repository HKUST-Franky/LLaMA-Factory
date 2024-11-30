# GLM-4-9B

**2024/08/12, The repository code has been updated and now requires `transformers>=4.44.0`. Please update your dependencies accordingly.**

## Model Introduction

GLM-4-9B is the open-source version of the latest generation of pre-trained models in the GLM-4 series launched by Zhipu
AI. In the evaluation of data sets in semantics, mathematics, reasoning, code, and knowledge, **GLM-4-9B**
and its human preference-aligned version **GLM-4-9B-Chat** have shown superior performance beyond Llama-3-8B. In
addition to multi-round conversations, GLM-4-9B-Chat also has advanced features such as web browsing, code execution,
custom tool calls (Function Call), and long text
reasoning (supporting up to 128K context). This generation of models has added multi-language support, supporting 26
languages including Japanese, Korean, and German. We have also launched the **GLM-4-9B-Chat-1M** model that supports 1M
context length (about 2 million Chinese characters) and the multimodal model GLM-4V-9B based on GLM-4-9B.
**GLM-4V-9B** possesses dialogue capabilities in both Chinese and English at a high resolution of 1120*1120.
In various multimodal evaluations, including comprehensive abilities in Chinese and English, perception & reasoning,
text recognition, and chart understanding, GLM-4V-9B demonstrates superior performance compared to
GPT-4-turbo-2024-04-09, Gemini 1.0 Pro, Qwen-VL-Max, and Claude 3 Opus.

We evaluated the GLM-4-9B base model on some typical tasks, and the results are as follows:

| Model               |   MMLU   |  C-Eval  |   GPQA   |  GSM8K   |   MATH   | HumanEval |
|:--------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:---------:|
| Llama-3-8B          |   66.6   |   51.2   |    -     |   45.8   |    -     |     -     | 
| Llama-3-8B-Instruct |   68.4   |   51.3   |   34.2   |   79.6   |   30.0   |   62.2    |
| ChatGLM3-6B-Base    |   61.4   |   69.0   |    -     |   72.3   |   25.7   |     -     |
| GLM-4-9B            | **74.7** | **77.1** | **34.3** | **84.0** | **30.4** | **70.1**  |

**This repository is the base version of GLM-4-9B, supporting 8K context length.**

For more inference code and requirements, please visit our [github page](https://github.com/THUDM/GLM-4).

## LICENSE

The weights of the GLM-4 model are available under the terms of [LICENSE](LICENSE).


## Citations

If you find our work useful, please consider citing the following paper.

```
@misc{glm2024chatglm,
      title={ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools}, 
      author={Team GLM and Aohan Zeng and Bin Xu and Bowen Wang and Chenhui Zhang and Da Yin and Diego Rojas and Guanyu Feng and Hanlin Zhao and Hanyu Lai and Hao Yu and Hongning Wang and Jiadai Sun and Jiajie Zhang and Jiale Cheng and Jiayi Gui and Jie Tang and Jing Zhang and Juanzi Li and Lei Zhao and Lindong Wu and Lucen Zhong and Mingdao Liu and Minlie Huang and Peng Zhang and Qinkai Zheng and Rui Lu and Shuaiqi Duan and Shudan Zhang and Shulin Cao and Shuxun Yang and Weng Lam Tam and Wenyi Zhao and Xiao Liu and Xiao Xia and Xiaohan Zhang and Xiaotao Gu and Xin Lv and Xinghan Liu and Xinyi Liu and Xinyue Yang and Xixuan Song and Xunkai Zhang and Yifan An and Yifan Xu and Yilin Niu and Yuantao Yang and Yueyan Li and Yushi Bai and Yuxiao Dong and Zehan Qi and Zhaoyu Wang and Zhen Yang and Zhengxiao Du and Zhenyu Hou and Zihan Wang},
      year={2024},
      eprint={2406.12793},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```
