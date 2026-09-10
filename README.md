# SOTANER — Reproduksi berbasis Notebook

Reproduksi eksperimen paper **"What do we really know about State of the art NER?"**
(Vajjala & Balasubramaniam, LREC 2022). Fungsi berat ada di modul `helpers/` (dipakai
bersama oleh kedua notebook).

| Notebook | Untuk | Data | Device |
|---|---|---|---|
| [`SOTANER_Reproduction.ipynb`](SOTANER_Reproduction.ipynb) | Windows + conda env `sotaner` (lokal) | `../conll-2012/v4/` LDC lokal | GPU 4 GB → banyak proses di CPU |
| [`SOTANER_Reproduction_Colab.ipynb`](SOTANER_Reproduction_Colab.ipynb) | **Google Colab** | **HuggingFace** `conll2012_ontonotesv5/english_v4` | **semua model di GPU** |

Isi keduanya setara — 3 section: Persiapan+Tabel 2 · Black-box Tabel 3–6 · Training
Tabel 7–8 + Figure 1. Setiap tabel hasil punya kolom **Reported on Paper**,
**Obtained** (hasil kita), **Delta** (`Obtained − Reported`). Output:
`results/SOTANER_reproduction_tables.xlsx` (+ CSV per tabel) + `results/Figure1_cross_genre.png`.

`helpers/*.py` diadaptasi minimal dari `../SOTANER Windows/`; yang benar-benar baru:
`helpers/conll_to_bio.py` (pembaca `conll-2012/v4`), `helpers/hf_to_bio.py` (loader
HuggingFace — output **byte-identik** dengan `conll_to_bio`), `helpers/paper_values.py`.

---

## 0. Menjalankan di Google Colab (`SOTANER_Reproduction_Colab.ipynb`)

1. Runtime GPU: **Runtime → Change runtime type → GPU** (mis. A100 / L4 / T4).
2. Upload folder ini (minimal `helpers/` + `config.cfg`) ke Google Drive di
   `MyDrive/SOTANER Notebook Based/`. (Cukup 2 item itu; folder `data/`, `models/`,
   `results/` dibuat otomatis di `MyDrive/SOTANER Notebook Based/colab_run/`.)
3. Buka `SOTANER_Reproduction_Colab.ipynb` di Colab, jalankan Section 0 (install + mount
   Drive; sesuaikan `PROJECT_DIR` bila nama folder berbeda), lalu lanjut berurutan.
4. Knob di sel 0.4: `FULL_RUN` (False = cek alur cepat; True = skala paper),
   `RUN_SPARKNLP`, `RESUME` (lanjut dari fold yang sudah selesai bila runtime putus).

Semua **inferensi & training model** jalan di GPU (spaCy `--gpu-id 0`, Stanza
`use_gpu=True`, Spark NLP `sparknlp.start(gpu=True)` — NerDL **best-effort**, bisa jatuh
ke CPU bila TF/CUDA tak cocok; cek `nvidia-smi` saat sel training jalan). Prep data
(HF→BIO, split, perturbasi) tetap CPU (ringan). Output persist di Drive; `RESUME=True`
melanjutkan lintas sesi.

---

## 1. Environment

Pakai conda env **`sotaner`** (sama dengan `../SOTANER Windows/`):

```powershell
conda activate sotaner
cd "D:\OneDrive\Penelitian\NER Methods Comparison\SOTANER Notebook Based"
# paket inti sudah terpasang di env; tambahan yang dipakai notebook:
python -m pip install matplotlib openpyxl
```

Versi yang diverifikasi: Python 3.12.9 · spacy 3.8.11 (`en_core_web_trf` 3.8.0) ·
stanza 1.14.0 · pyspark 3.5.9 · spark-nlp 5.5.3 · seqeval 1.2.2 · scikit-learn 1.8 ·
scipy 1.17 · faker · pandas 3.0.

Jalankan Jupyter/VS Code dari folder ini sehingga `Path.cwd()` = `SOTANER Notebook Based`.

## 2. Dataset

Notebook Section 1 membaca:

```
../conll-2012/v4/data/{train,development,test}/data/english/annotations/<genre>/**/*.v4_gold_conll
```

dan menulis BIO 4 kolom ke `data/bio/`. Validasi otomatis: `test/onto.all6.ner` harus
**8.262 kalimat / 11.257 entitas** (= support micro Tabel 3 paper). `pt` (Alkitab) tak
punya anotasi NE → dikecualikan.

## 3. Spark NLP (opsional — set `RUN_SPARKNLP=False` untuk melewati)

Butuh setup Windows sekali jalan (sama seperti `../SOTANER Windows/DOCUMENTATION.md` §2):

- **JDK 17** di `C:\Program Files\Java\jdk-17` (override via env `JAVA_HOME`).
- `C:\hadoop\bin\` berisi **`winutils.exe` + `hadoop.dll`** (Hadoop 3.3.x); override via `HADOOP_HOME`.
- Model TF offline di `~/cache_pretrained/` :
  ```powershell
  curl -L -o bert.zip "https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_cased_en_2.6.0_2.4_1598340336670.zip"
  curl -L -o onto.zip "https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_bert_base_cased_en_2.7.0_2.4_1607197077494.zip"
  Expand-Archive bert.zip "$env:USERPROFILE\cache_pretrained\bert_base_cased"
  Expand-Archive onto.zip "$env:USERPROFILE\cache_pretrained\onto_bert_base_cased"
  ```
  Di **Colab** langkah ini otomatis: `helpers.spark_session.stage_pretrained_models()`
  (dipanggil di sel Spark NLP notebook, juga sebagai fallback di `load_bert`/`load_nerdl`)
  mengunduh + meng-unzip dua edisi TF yang sama ke `/root/cache_pretrained/`. Perlu karena
  `BertEmbeddings.pretrained("bert_base_cased")` di spark-nlp 5.5.3 keliru resolve ke
  `DistilBertForTokenClassification` → `ClassCastException`.
- Jar assembly offline: `helpers.spark_session` otomatis memakai `../SOTANER Windows/jars/spark-nlp-assembly-5.5.3.jar`.
  Bila folder ini dipindah keluar dari repo, buat `jars/` di sini dan salin jar itu ke dalamnya
  (atau set env `SOTANER_SPARKNLP_JAR` ke path jar).

`helpers.spark_session.start_spark()` menyetel `JAVA_HOME`/`HADOOP_HOME`/`PATH`/`SPARK_HOME`/
`SPARK_LOCAL_IP` secara otomatis. Error `ShutdownHookManager: Failed to delete ... spark-nlp-assembly.jar`
saat `spark.stop()` **tidak berbahaya** (lag lock file Windows).

## 4. Menjalankan

| Section | Beban | Catatan |
|---|---|---|
| **1** Persiapan + Tabel 2 | sedang | build BIO ~2 mnt; eval `en_core_web_trf` di CPU ~35–55 mnt untuk test penuh |
| **2** Black-box (Tabel 3–6) | sedang–berat | Tabel 6 = 6× pass test penuh (spaCy+Stanza ~4–5 jam; Spark NLP ~30–45 mnt) |
| **3** Training (Tabel 7–8, Fig 1) | **berat** | atur `FULL_RUN` |

**Sel konfigurasi** (di atas Section 1):

```python
FULL_RUN     = False   # False: 3 fold + max_steps kecil (cek alur) ; True: 10 fold skala paper
RUN_SPARKNLP = True
SEED         = 42
```

`FULL_RUN=False` memastikan seluruh alur jalan dalam hitungan menit per library; **angka
Tabel 7 skala penuh hanya valid dengan `FULL_RUN=True`** (sesi berjam-jam). Tiap library
punya sel training sendiri agar bisa dijalankan bertahap. Untuk latih di GPU: set env
`SOTANER_SPACY_GPU=1` (spaCy) atau ubah `STANZA_GPU=True` di sel 3.2b.

## 5. Deviasi dari paper

1. **Data** `conll-2012/v4` `.v4_gold_conll` LDC lokal (divalidasi ke 11.257 entitas test).
2. **spaCy retraining** (Tabel 7/8/Fig 1) memakai `config.cfg` rilisan penulis = **CNN
   tok2vec transition-based parser**, bukan fine-tune `en_core_web_trf`. Evaluasi
   off-the-shelf (Tabel 2–6) tetap `en_core_web_trf`.
3. **Stanza** 1.14 (`stanza.models.ner_tagger`, pretrain `conll17.pt`); paper ~1.4 + `combined.pt`.
   CLI stanza 1.14 sudah tak punya `--lang`/`--output_file` → helper memakai `--shorthand`
   + `--eval_output_file` (TSV BIOES) lalu di-skor ulang dengan seqeval.
4. **Spark NLP** 5.5.3 + model TF `onto_bert_base_cased`/`bert_base_cased` (era 2020, sesuai
   paper); paper menyebut 3.1.2. `NerDLApproach.maxEpochs` tak disebut paper → 1 (cek) / 10 (FULL_RUN).
5. **Tabel 2 "Reported on Paper"** = kolom *Obtained* paper; angka situs library jadi kolom catatan.
6. **Random split** = `KFold(10, shuffle=True, random_state=42)` + `ShuffleSplit` dev 10%
   (proporsi ~81/9/10) — tafsiran reproducible dari "10 random splits, proporsi = standard split".
7. Inkonsistensi internal paper yang diketahui: spaCy `bc` Tabel 4 (91.55) vs Tabel 5 (88.72);
   spaCy `GPE` Tabel 3 (95.36) vs Tabel 6 baris None (95.61).

## 6. Isi `helpers/`

Semua modul **lintas-platform** (Windows lokal + Linux/Colab). Dipakai oleh kedua notebook.

| Modul | Dari | Fungsi utama |
|---|---|---|
| `conll_to_bio.py` | **baru** (logika bracket→BIO dari `SOTANER/conll-to-bio.py`) | `build_all(v4_root, out_dir)` — data `conll-2012/v4` lokal |
| `hf_to_bio.py` | **baru** (dari `SOTANER Windows/hf_to_bio.py`) | `build_all(out_dir, config="english_v4")` — dari HuggingFace; output **byte-identik** dengan `conll_to_bio` |
| `bio_utils.py` | baru (util bersama) | `read_bio_file`, `read_sentence_blocks`, `write_bio`, `count_entities`, `bioes_to_bio`, konstanta genre/tipe |
| `paper_values.py` | ekstraksi paper + `Results-Details.xlsx` | dict Tabel 2–8 + Figure 1 |
| `eval_blackbox.py` | `evaluation_spacy_stanza.py` + `run_blackbox.py` | `load_spacy(gpu=None)`, `load_stanza(use_gpu=True)`, `evaluate_bio`, `run_blackbox` |
| `spark_session.py` | `spark_start.py` + `spark_env.ps1` | `start_spark(gpu=False)` — branch Windows (offline jar) / Linux (`sparknlp.start`); `load_bert`, `load_nerdl` |
| `eval_sparknlp.py` | `evaluation_sparknlp.py` | `build_pipeline`, `evaluate_bio_sparknlp` |
| `perturb.py` | `input_perturbances_faker.py` + `make_perturbations.py` | `perturb_file`, `make_all_perturbations` |
| `splits.py` | `generate_splits.py` | `make_kfold_splits` |
| `crossgenre.py` | `make_crossgenre_sets.py` | `make_sets` |
| `train_spacy.py` | `spacy-stanza-scripts/*_ner_spacy.py` | `prep`, `train(gpu_id=)`, `evaluate` |
| `train_stanza.py` | `spacy-stanza-scripts/*_ner_stanza.py` | `bio_to_json`, `train(use_gpu=, batch_size=)`, `predict_and_score(use_gpu=)`, `find_wordvec_pretrain` |
| `train_sparknlp.py` | `train_sparknlp.py` | `fit_nerdl`, `score_nerdl`, `run_fold` |
| `report_utils.py` | ide dari `collect_results.py` | `comparison_table`, `summarise_folds`, `export_all`, `f1_of`, `micro_f1_from_text` |
