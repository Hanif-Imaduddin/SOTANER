"""Shared Spark NLP session + model loading.

Two code paths:

* **Windows** (``os.name == "nt"``) -- the battle-tested local setup from
  ``SOTANER Windows/spark_start.py``. See ``SOTANER Windows/DOCUMENTATION.md``
  Section 4 for the symptom -> cause -> fix table: stale ``SPARK_HOME``, an
  offline ``spark-nlp-assembly`` jar (Ivy/Maven hangs), JavaSerializer + loopback
  + JDK-17 ``--add-opens`` for the ``idWithoutTopologyInfo is null`` NPE. Needs
  the one-time offline setup in DOCUMENTATION.md Section 2c (JDK 17,
  ``C:\\hadoop\\bin`` with ``winutils.exe`` + ``hadoop.dll``, the assembly jar,
  the TF model editions under ``~/cache_pretrained/``).

* **Linux / Colab** (anything else) -- a plain ``sparknlp.start(gpu=..., memory=...)``.
  Models come from ``.pretrained()`` (the JSL index works when there's internet);
  ``gpu=True`` pulls the ``spark-nlp-gpu`` assembly (NerDL/BERT on GPU is
  best-effort -- falls back to CPU if the CUDA/cuDNN build doesn't match).
"""
from __future__ import annotations

import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_NB_ROOT = os.path.dirname(_HERE)
_WIN_ROOT = os.path.join(os.path.dirname(_NB_ROOT), "SOTANER Windows")

_ADD_OPENS = " ".join(
    "--add-opens=java.base/{}=ALL-UNNAMED".format(p) for p in [
        "java.lang", "java.lang.invoke", "java.lang.reflect", "java.io", "java.net",
        "java.nio", "java.util", "java.util.concurrent", "java.util.concurrent.atomic",
        "sun.nio.ch", "sun.nio.cs", "sun.security.action", "sun.util.calendar",
    ]
) + " -Djdk.reflect.useDirectMethodHandle=false"


def prepare_env(java_home=r"C:\Program Files\Java\jdk-17", hadoop_home=r"C:\hadoop",
                verbose=True):
    """Idempotently set the Windows env vars Spark NLP needs (what
    ``SOTANER Windows/spark_env.ps1`` did before launch), so the notebook can
    start Spark without a pre-configured shell. Override paths via the env vars
    ``JAVA_HOME`` / ``HADOOP_HOME`` before calling, or pass args here."""
    java_home = os.environ.get("JAVA_HOME") or java_home
    hadoop_home = os.environ.get("HADOOP_HOME") or hadoop_home
    os.environ["JAVA_HOME"] = java_home
    os.environ["HADOOP_HOME"] = hadoop_home
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    add = os.pathsep.join([os.path.join(java_home, "bin"),
                           os.path.join(hadoop_home, "bin")])
    if add not in os.environ.get("PATH", ""):
        os.environ["PATH"] = add + os.pathsep + os.environ.get("PATH", "")

    # a stale system SPARK_HOME (e.g. C:\spark) breaks pyspark's gateway launch
    try:
        import pyspark
        os.environ["SPARK_HOME"] = os.path.dirname(pyspark.__file__)
    except Exception:  # noqa: BLE001
        pass
    os.environ.pop("PYSPARK_SUBMIT_ARGS", None)

    if verbose:
        ok_java = os.path.isdir(os.path.join(java_home, "bin"))
        ok_hadoop = os.path.exists(os.path.join(hadoop_home, "bin", "hadoop.dll"))
        print(f"spark env: JAVA_HOME={java_home} ({'ok' if ok_java else 'MISSING'})  "
              f"HADOOP_HOME={hadoop_home} (hadoop.dll {'ok' if ok_hadoop else 'MISSING'})")
    return java_home, hadoop_home


def _assembly_jar():
    env = os.environ.get("SOTANER_SPARKNLP_JAR")
    if env and os.path.exists(env):
        return env
    hits = []
    for root in (os.path.join(_NB_ROOT, "jars"), os.path.join(_WIN_ROOT, "jars")):
        hits += sorted(glob.glob(os.path.join(root, "spark-nlp-assembly-*.jar")))
    if not hits:
        return None
    try:
        import sparknlp
        ver = sparknlp.version()
        for h in hits:
            if ("assembly-" + ver + ".jar") in os.path.basename(h):
                return h
    except Exception:  # noqa: BLE001
        pass
    return hits[0]


def start_spark(memory="8g", gpu=False, prep_env=True):
    """Return a SparkSession with Spark NLP ready.

    ``gpu`` only affects the Linux/Colab path (loads ``spark-nlp-gpu``). On
    Windows it is ignored (the offline TF assembly is CPU).
    """
    if os.name != "nt":
        import sparknlp
        print(f"Spark NLP: sparknlp.start(gpu={gpu}, memory={memory!r})")
        return sparknlp.start(gpu=gpu, memory=memory)

    if prep_env:
        prepare_env()
    from pyspark.sql import SparkSession

    b = (
        SparkSession.builder.appName("sotaner-nb").master("local[*]")
        .config("spark.driver.memory", memory)
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
        .config("spark.driver.maxResultSize", "0")
        .config("spark.driver.extraJavaOptions", _ADD_OPENS)
        .config("spark.executor.extraJavaOptions", _ADD_OPENS)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    )
    jar = _assembly_jar()
    if jar:
        b = b.config("spark.jars", jar.replace("\\", "/"))
        print("Spark: offline jar", os.path.basename(jar))
    else:
        b = b.config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.3")
        print("Spark: resolving spark-nlp from Maven (no offline jar found)")
    return b.getOrCreate()


# --- pretrained model loading (from disk; .pretrained() hangs on the JSL index here)
_CACHE = os.path.expanduser("~/cache_pretrained")
_LOCAL_ROOTS = (os.path.join(_NB_ROOT, "models"), os.path.join(_WIN_ROOT, "models"), _CACHE)

_JSL_S3 = "https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/"
_OFFLINE_ZIPS = {
    # name -> exact TF edition used by the validated Windows run (DOCUMENTATION.md section 2)
    "bert_base_cased":      "bert_base_cased_en_2.6.0_2.4_1598340336670.zip",
    "onto_bert_base_cased": "onto_bert_base_cased_en_2.7.0_2.4_1607197077494.zip",
}


def stage_pretrained_models(names=None, cache_dir=None, verbose=True):
    """Download + unzip the exact TF model editions the paper/Windows run used into
    ``~/cache_pretrained/<name>/`` so ``load_bert`` / ``load_nerdl`` take the offline
    path.

    Works around spark-nlp 5.5.3 resolving ``bert_base_cased`` (and possibly
    ``onto_bert_base_cased``) to the wrong artifact via ``.pretrained()`` -- a
    ``DistilBertForTokenClassification`` that then fails to deserialize as
    ``BertEmbeddings`` (``ClassCastException``). No-op for any name already staged.
    Returns the cache dir.
    """
    import urllib.request
    import zipfile

    cache_dir = cache_dir or _CACHE
    names = list(names) if names else list(_OFFLINE_ZIPS)
    os.makedirs(cache_dir, exist_ok=True)
    for name in names:
        dest = os.path.join(cache_dir, name)
        if os.path.isdir(dest) and os.listdir(dest):
            if verbose:
                print(f"  {name}: already staged ({dest})")
            continue
        url = _JSL_S3 + _OFFLINE_ZIPS[name]
        zpath = os.path.join(cache_dir, _OFFLINE_ZIPS[name])
        if verbose:
            print(f"  {name}: downloading {url}")
        urllib.request.urlretrieve(url, zpath)
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(dest)  # JSL zips are flat -> matches Windows Expand-Archive
        os.remove(zpath)
        if verbose:
            print(f"  {name}: staged -> {dest}")
    return cache_dir


def _uri(p):
    return "file:///" + os.path.abspath(p).replace("\\", "/")


def _find_local(name):
    for root in _LOCAL_ROOTS:
        p = os.path.join(root, name)
        if os.path.isdir(p) and os.listdir(p):
            return _uri(p)
        hits = [h for h in sorted(glob.glob(os.path.join(root, name + "_en_*")))
                if os.path.isdir(h) and os.listdir(h)]
        if hits:
            return _uri(hits[-1])
    return None


def _local_or_staged(name):
    """Local model URI if present; otherwise try staging a known-good offline
    edition and re-check. Returns ``None`` to let the caller fall back to
    ``.pretrained()``."""
    local = _find_local(name)
    if local or name not in _OFFLINE_ZIPS:
        return local
    try:
        stage_pretrained_models([name])
    except Exception as e:  # noqa: BLE001 - fall through to .pretrained()
        print(f"stage_pretrained_models({name!r}) failed: {e}")
    return _find_local(name)


def load_bert(name="bert_base_cased"):
    from sparknlp.annotator import BertEmbeddings
    local = _local_or_staged(name)
    if local:
        print("BertEmbeddings.load(%s)" % local)
        return BertEmbeddings.load(local)
    print("BertEmbeddings.pretrained(%s)" % name)
    return BertEmbeddings.pretrained(name, "en")


def load_nerdl(name="onto_bert_base_cased"):
    from sparknlp.annotator import NerDLModel
    local = _local_or_staged(name)
    if local:
        print("NerDLModel.load(%s)" % local)
        return NerDLModel.load(local)
    print("NerDLModel.pretrained(%s)" % name)
    return NerDLModel.pretrained(name, "en")
