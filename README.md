# 🖼️ 皮肤癌分类项目

本项目基于 Vision Transformer (ViT) 模型对皮肤镜图像进行分类。

---

## 📂 数据集准备 (HAM10000)

**注意：** 为了方便上传，本仓库中仅保留了 `24` 张示例图片。如需复现完整训练，请按以下步骤操作：

1.  **下载数据集**
    请前往 Kaggle 下载完整的 HAM10000 数据集：
    [https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

2.  **整理文件夹结构**
    下载完成后，请对文件夹进行如下处理：
    *   **保留：** 请保留 `part_1` 和 `part_2` 两个文件夹。
        *   *(原名为 `HAM10000_images_part_1` 和 `HAM10000_images_part_2`，请重命名为 `part_1` 和 `part_2`)*
    *   **删除：** 请删除其余的 `.csv` 文件（共 5 个）。

3.  **覆盖目录**
    将整理好的 `part_1` 和 `part_2` 文件夹替换本项目目录下的对应文件夹。

---

## 🤖 模型准备

*   **模型名称：** `google/vit-base-patch16-224`
*   **说明：** 由于模型文件较大，未上传至仓库。
*   **操作：** 复现时，请前往 Hugging Face 或其他方式自行下载该模型，并将其保存在项目根目录下的 `vit-base-patch16-224` 文件夹中。

---

## ⚙️ 环境与运行

1.  **安装依赖**
    请确保已安装所需的 Python 库。若缺少库，请使用 `pip` 自行下载安装。

2.  **修改路径**
    在运行前，请打开 `train.py` 文件，将其中涉及的文件路径修改为您的本地实际路径。

3.  **启动训练**
    完成上述配置后，直接运行以下命令即可开始训练：
    ```bash
    python train.py
    ```
