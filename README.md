# RefundProof

**Validating AI-Assisted Image Screening for Digital Evidence: A Framework for Trustworthy Refund Forensics**

**Usama Zafar, Fan Wu and Yin-Chia Hsieh**

Code and numerical records for controlled image alteration, frozen-detector evaluation, source-group calibration and operating-policy analysis. The main cohort contains 462 images from 182 source/product groups. The external challenge contains 5,590 FraudBench images.

## Contents

| Folder | Contents |
|---|---|
| `code/` | Dataset construction, detector evaluation and statistical analysis |
| `data/` | Four consolidated CSV files and a data dictionary |
| `config/` | Protocol, environment, source-group partitions and fitted-head records |
| `fitted_heads/` | 36 fitted score-head and calibration bundles |

## Evaluation

Install the Python dependencies in a compatible CUDA environment:

```bash
python -m pip install -r requirements.txt
```

Run `code/evaluate_detectors.py` in Kaggle with a GPU. Attach the retained study media, computational masks, `primary_manifest.csv`, `manifest_all.csv` and `study_configuration.json`. These can be supplied in `RefundProof_Study_Inputs.zip` under `RefundProof_PAPER/`. The script also reads the external manifest and saved comparator probabilities. Store download credentials in a Kaggle `HF_TOKEN` secret. Released checkpoint downloads require Internet access.

The evaluation checks image identities, checkpoint hashes and the detector repository revision. Source groups separate fitting, calibration and testing. The script exports final probabilities, discrimination and calibration metrics, paired comparisons, regional measurements, JPEG sensitivity and operating policies.

`code/construct_dataset.py` implements object-grounded editing and RGB/residual model training through `code/study_pipeline.py`. `evaluation_core.py`, `legacy_features.py` and `region_audit.py` supply the numerical methods. Dataset construction requires the retained source package and the `RP_REFERENCE` directory used by the construction protocol.

## Numerical records

`data/data_dictionary.md` describes the four CSV files. `config/data_schema.json` maps consolidated fields to the analysis records and includes category accounting and mask distributions. Numerical values retain their saved precision.

Each object in `fitted_heads/` contains a fitted logistic head and calibrator. Filenames identify the checkpoint, head type, seed and fold. `config/fitted_head_records.json` records object hashes, group assignments, calibration scores and held-out predictions. `config/environment.json` records the study environment. Load the fitted objects with compatible library versions.

## Source data and checkpoints

Source photographs, customer review text and backbone weights are obtained from their original providers under the applicable terms. FraudBench is available from [TristanYan/FraudBench](https://huggingface.co/datasets/TristanYan/FraudBench), revision `27d94c1a4c4b087845f8a8e916473cdd6a5c6011`. Amazon review-derived source material is described on the [UC San Diego dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/). ForgeLens code and released checkpoints are available from the [original authors](https://github.com/Yingjian-Chen/ForgeLens). Exact checkpoint identities and input procedures are recorded in `config/protocol.json`.

## Citation and license

`CITATION.cff` contains the author and archive metadata. The software uses the [MIT license](LICENSE). Source datasets and released detector models retain their own terms.
