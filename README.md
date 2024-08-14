# 知心智医 CardioRAG

## 项目背景
近年来，心血管疾病的发病率不断上升，医疗资源的有限性使人民群众的看病压力逐年增加。由于缺乏相关专业知识，患者对于疾病和检查报告、用药以及不同医院专精科室的认识不足，从而对疾病的治疗产生影响。

## 目标与目标市场
「知心智医」产品面向广大心血管疾病患者及家属，旨在为用户提供专业的心血管疾病问答，作为一个人工智能的「私人医生」。

## 运行要求

1. 克隆项目代码

    ```shell
    git clone https://github.com/peylix/CardioRAG.git
    ```

2. 安装所有要求文件

    ```shell
    pip install -r requirements.txt
    ```

3. 【可选】可以在 `rag_core.py` 的 `preprocess` 函数中指定本地模型路径

4. 运行项目

    ```shell
    streamlit run chatbot_ui.py
    ```
