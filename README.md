# ComfyUI PIC Pack

这是一个用于集中维护个人 ComfyUI 节点的节点包。当前已整理进来的功能子包是 `pic_pack/anima`，用于 Anima 提示词生成辅助。

## 当前结构

```text
comfyui-PIC-Pack/
├── __init__.py
├── README.md
└── pic_pack/
    └── anima/
        ├── nodes.py
        ├── data/
        │   ├── cooccurrence_pmi.json
        │   └── character_features.json
        └── systemprompt/
            └── Anima_Core_Rules.txt
```

根目录的 `__init__.py` 负责向 ComfyUI 暴露节点映射；具体节点实现放在功能子包中，后续新增节点也按这个方式归类。ComfyUI 左侧节点菜单通过节点类的 `CATEGORY` 分组，当前 Anima 节点显示在 `PIC Pack/Anima` 下。

## Anima 子包

`pic_pack/anima` 当前提供：

- `PIC_AnimaPMIExpand`
- `PIC_AnimaRAGSearch`，兼容旧工作流 ID
- `PIC_AnimaPromptAssembler`
- `PIC_GGUFTextLLM`
- `PIC_StyleDirector`

运行时数据从 `pic_pack/anima/data/` 读取，System Prompt 从 `pic_pack/anima/systemprompt/` 读取。
