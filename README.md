<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill LLaVA Module

This repository contains the code supporting the LLaVA base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[LLaVA](https://github.com/haotian-liu/LLaVA) is a multi-modal language model with object detection capabilities.  You can use LLaVA with autodistill for object detection. [Learn more about LLaVA 1.5](https://blog.roboflow.com/first-impressions-with-llava-1-5/), the most recent version of LLaVA at the time of releasing this package.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [LLaVA Autodistill documentation](https://autodistill.github.io/autodistill/base_models/llava/).

## Installation

To use CLIP with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-clip
```

## Quickstart

```python
from autodistill_llava import LLaVA

# define an ontology to map class names to our LLaVA prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = LLaVA(
    ontology=CaptionOntology(
        {
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpeg")
```


## License

This model is licensed under an [Apache 2.0 License](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!