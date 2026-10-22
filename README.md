# signal-or-noise

A small learning project: **should an event count as signal, or as noise?**

Radiation detectors produce many events, some coming from the object of interest and some
from background. A common approach is a single hand-tuned threshold. This project compares
that rule with a machine-learning classifier and shows how both behave as noise grows.

> Interactive Streamlit app comparing a fixed-threshold rule with machine learning for
> selecting signal events from detector pulses (neutron/gamma data), and showing how noise
> breaks the rule.

## What it does

1. Uses real neutron/gamma detector pulses (public Zenodo dataset).
2. Builds a classic one-feature rule (tail-to-total ratio) and traces its ROC trade-off.
3. Trains a logistic-regression model on all 280 samples of each pulse and compares it with
   the rule while synthetic Gaussian noise is added to the test pulses.
4. Wraps this in a Streamlit app with a noise slider and a threshold slider.

Both methods are trained on clean training data only. The noise is synthetic, so the results
illustrate an idea; they say nothing about any specific real detector system.

## Roadmap

- [x] Project setup, data download, data loader
- [x] Look at the raw pulses
- [x] Classic tail-to-total rule and ROC curve
- [x] Machine-learning classifier, compared with the rule under added noise
- [x] Streamlit app (first version)
- [ ] Synthetic 2D scene and side-by-side heat maps
- [ ] Publish on GitHub

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The app needs two small files:

- `models/model.pkl`: a pickled dict `{"model": <trained model>, "thr": <threshold>}`
- `sample/test_sample.npz`: arrays `X` (n, 280, float32) and `y` (n, 0 = gamma, 1 = neutron)

They are created in the analysis notebook from the public dataset (see Data). Then run:

```bash
streamlit run app.py
```

`requirements.txt` pins `scikit-learn` because pickle files are version-sensitive; use the
same version that created `models/model.pkl`. Only unpickle files you created yourself.

To explore the raw data locally (about 78 MB), download and unzip it into `data/raw`:

```bash
mkdir -p data
curl -L -o data/Datasets.zip "https://zenodo.org/records/16730902/files/Datasets.zip?download=1"
python -c "import zipfile; zipfile.ZipFile('data/Datasets.zip').extractall('data/raw')"
python src/step2_peek.py
```

## Repository layout

```
app.py          Streamlit app
models/         trained model (pickle)
sample/         small test sample used by the app
notebooks/      step-by-step notebooks
src/            data loading and exploration code
data/           downloaded data (git-ignored)
outputs/        generated figures (git-ignored)
```

## Data and citation

Liu et al., "Pulse Shape Discrimination Algorithms: A Survey and Benchmark", Zenodo,
doi:10.5281/zenodo.16730902 (EJ299-33 scintillator, AmBe source), released under the MIT
license. Each row is one pulse of 280 samples; gamma and neutron events are stored in
separate files, so labels come from the file names. The small test sample in `sample/` is
a random subset of that dataset.

The authors ask that anyone using any part of this work cites both the original paper and
the Zenodo dataset.

## Limitations

- The noise is synthetic Gaussian noise added to the pulses.
- The classic rule is deliberately simple; a better hand-built rule would also resist noise
  better. The point is that weights learned from data beat equal weights, not that machine
  learning is magic.
- The planned heat-map step will use a synthetic scene for illustration only.
