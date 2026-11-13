# Tortuosité rétine

A Streamlit app that measures how tortuous (twisty) the blood vessels are in retinal fundus images.

It segments the arteries and veins of an eye, lets you review and correct the detected vessels, then computes a tortuosity score per vessel and per eye. Results can be exported as CSV/Excel files and PDF reports.

![Screenshot of the app](docs/screenshot.png)

## Getting started

You need [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
uv run streamlit run tortuosite_score/app/app.py
```

The app opens in your browser.

To use the deep-learning segmentation backends (DCP / VascX), install the extra dependencies instead:

```bash
uv sync --extra deep
```

## Credits

This project builds on the work of others:

- **VascX** — artery/vein and optic disc segmentation models by the Eyened group ([Eyened/vascx](https://huggingface.co/Eyened/vascx) on Hugging Face), loaded through the [vascx-simplify](https://pypi.org/project/vascx-simplify/) package.
- **DCP** — alternative vessel segmentation model by AIMC Lab, Renmin University of China ([ruc-aimc-lab/dcp](https://github.com/ruc-aimc-lab/dcp), checkpoint [AIMClab-RUC/UNet_DCP_1024](https://huggingface.co/AIMClab-RUC/UNet_DCP_1024)).
- **Toolkit — scoring methods** — the methods prefixed with **Toolkit —** in the app come from [coconetlero/tortuosity_metrics_python](https://github.com/coconetlero/tortuosity_metrics_python).
- **Grisan and Tortuosity Density scores** — based on Grisan E, Foracchia M, Ruggeri A. "A Novel Method for the Automatic Grading of Retinal Vessel Tortuosity." *IEEE Transactions on Medical Imaging* 27(3), 2008.
- **[skan](https://github.com/jni/skan)** — skeleton analysis library used to trace vessel centerlines.

The scoring approach was also informed by:

- Ramos L, Novo J, Rouco J, Romeo SJ, Ortega M. "[Computational assessment of the retinal vascular tortuosity integrating domain-related information](https://www.nature.com/articles/s41598-019-56507-7)." *Scientific Reports*, 2019.
- Hervella AS, Rouco J, Novo J, Ortega M. "[Explainable artificial intelligence for the automated assessment of the retinal vascular tortuosity](https://link.springer.com/article/10.1007/s11517-023-02978-w)." *Medical & Biological Engineering & Computing*, 2024.
- Ramos L, Novo J, Rouco J, Ortega M. "[Retinal vascular tortuosity assessment: inter-intra expert analysis and correlation with computational measurements](https://link.springer.com/article/10.1186/s12874-018-0598-3)." *BMC Medical Research Methodology*, 2018.
