"""
tpf_study.py : complete code of the study
"Artificial intelligence in financial forecasting and participatory data analysis:
 the case of Moroccan participatory banks (monthly data, Oct 2018 - Jun 2026)".

Everything that produced the results of the thesis is in this single file:
  - data loading, IQR outlier rule, random / chronological splits
  - exploratory statistics: descriptive, ADF, VIF, PCA, bivariate relations, variable selection
  - generative models: VAE and conditional GAN (trained on the training set only)
  - the eight predictors with the hyper-parameters retained in the thesis (LR, DT, RF, XGB, MLP, SVR, GBR, ET)
    + Poisson/PPML benchmark
  - metrics (MAE, MSE, RMSE, R2, MAPE, Explained Variance, MedianAE)
  - tests: Diebold-Mariano, Wilcoxon, Friedman; SHAP; learning curve; sensitivity analysis
  - hypothesis tests H1-H3

Usage (from the repository root):
    python src/tpf_study.py eda
    python src/tpf_study.py run_all --seeds 10
    python src/tpf_study.py analyze
    python src/tpf_study.py hypotheses
    python src/tpf_study.py all                      # everything (about 30-40 minutes on one CPU)
"""
import os, sys, json, pickle, time, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, learning_curve, KFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             explained_variance_score, median_absolute_error)
from xgboost import XGBRegressor
from scipy import stats
from scipy.stats import ks_2samp, skew, kurtosis
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

# ------------------------------------------------------------------ configuration
DATA = "data/Base_Donnees_Finale_Augmentee.xlsx"     # real data: sheet 'Donnees_Reelles' (93 rows)
OUT = "results"; WORK = "work"
TARGET = "TPF(Target )"
DROP = ["Obs_ID", "PBC"]            # PBC removed beforehand (near-identical to the target)
FEATSETS = {"ALL17": [],            # variables excluded in each specification
            "NOBAL": ["REF", "CF", "EF", "TF", "TA", "TL", "RC", "CCA", "ID"]}   # accounting-identity variables
MODELS = ["LR", "DT", "RF", "XGB", "MLP", "SVR", "GBR", "ET"]
GENS = ["VAE", "cGAN"]
FIG = None
BLUE, ORG, GRN, RED, GRY = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096"


def configure(data, out, work):
    global DATA, OUT, WORK, FIG
    DATA, OUT, WORK = data, out, work
    FIG = os.path.join(OUT, "figures")
    for d in (OUT, WORK, FIG): os.makedirs(d, exist_ok=True)


try:                                   # Arabic labels in figures (optional)
    import arabic_reshaper
    from bidi.algorithm import get_display
    def ar(s): return get_display(arabic_reshaper.reshape(s))
except Exception:
    def ar(s): return s


def _style():
    global plt, sns
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    globals()["plt"], globals()["sns"] = plt, sns


def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"{name}.png"), dpi=170, bbox_inches="tight"); plt.close()


def L(sc, fs):
    return pickle.load(open(os.path.join(WORK, f"run_{sc}_{fs}.pkl"), "rb"))


# ---------------------------------------------------------------- data
def load_real():
    df = pd.read_excel(DATA, "Donnees_Reelles").drop(columns=DROP)
    df.insert(0, "date", pd.date_range("2018-10-01", periods=len(df), freq="MS"))
    return df


def iqr_filter(df, cols):
    q1, q3 = df[cols].quantile(.25), df[cols].quantile(.75)
    iqr = q3 - q1
    mask = ((df[cols] < q1 - 1.5 * iqr) | (df[cols] > q3 + 1.5 * iqr))
    return df.loc[~mask.any(axis=1)].reset_index(drop=True), mask


def make_split(df, scenario, seed=0, use=None):
    """scenario = '<IQR|FULL>_<random|chrono>'.
    IQR  : article-exact outlier removal (Q1-1.5IQR / Q3+1.5IQR, any variable) before the split
    FULL : no removal (keeps the Covid-19 / 2022-inflation months)
    random: random 80/20 split with a seed (article, Sect. 2.3.5)
    chrono: first 80% of months for training, last 20% for test (forecasting protocol)"""
    clean, mode = scenario.split("_")
    cols = use if use is not None else [c for c in df.columns if c != "date"]
    d = iqr_filter(df, cols)[0] if clean == "IQR" else df.copy()
    n = len(d); ntr = int(round(.8 * n))
    if mode == "random":
        p = np.random.RandomState(1000 + seed).permutation(n)
        return d.iloc[p[:ntr]].reset_index(drop=True), d.iloc[p[ntr:]].reset_index(drop=True)
    return d.iloc[:ntr].reset_index(drop=True), d.iloc[ntr:].reset_index(drop=True)


# ---------------------------------------------------------------- generators
def train_vae(Z, n_out, seed, latent=6, hidden=32, epochs=800, beta=0.01):
    tf.keras.utils.set_random_seed(seed)
    d = Z.shape[1]
    inp = keras.Input((d,))
    h = keras.layers.Dense(hidden, activation="relu")(inp)
    mu = keras.layers.Dense(latent)(h); lv = keras.layers.Dense(latent)(h)
    dec = keras.Sequential([keras.layers.Dense(hidden, activation="relu"), keras.layers.Dense(d)])
    enc = keras.Model(inp, [mu, lv])
    opt = keras.optimizers.Adam(1e-3)
    X = tf.constant(Z, tf.float32)

    @tf.function
    def step():
        with tf.GradientTape() as t:
            m, l = enc(X)
            z = m + tf.exp(.5 * l) * tf.random.normal(tf.shape(m))
            rec = tf.reduce_mean(tf.square(X - dec(z)))
            kl = -.5 * tf.reduce_mean(tf.reduce_sum(1 + l - tf.square(m) - tf.exp(l), 1))
            loss = rec + beta * kl
        v = enc.trainable_variables + dec.trainable_variables
        opt.apply_gradients(zip(t.gradient(loss, v), v))
        return rec, kl
    hist = []
    for e in range(epochs):
        r, k = step(); hist.append((float(r), float(k)))
    return dec(tf.random.normal((n_out, latent))).numpy(), hist


def train_cgan(Z, n_out, seed, noise=8, ncls=4, epochs=4000, lam=10.0):
    tf.keras.utils.set_random_seed(seed)
    d = Z.shape[1]
    y = Z[:, 0]                                  # target is column 0
    edges = np.quantile(y, np.linspace(0, 1, ncls + 1)[1:-1])
    cls = np.digitize(y, edges)
    C = tf.one_hot(cls, ncls)
    X = tf.constant(Z, tf.float32)
    G = keras.Sequential([keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(64, activation="relu"), keras.layers.Dense(d)])
    D = keras.Sequential([keras.layers.Dense(64, activation="leaky_relu"),
                          keras.layers.Dense(32, activation="leaky_relu"), keras.layers.Dense(1)])
    og = keras.optimizers.Adam(1e-4, beta_1=.5); od = keras.optimizers.Adam(1e-4, beta_1=.5)
    bce = keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def step():
        z = tf.random.normal((len(Z), noise))
        with tf.GradientTape() as tg, tf.GradientTape() as td:
            fake = G(tf.concat([z, C], 1))
            dr = D(tf.concat([X, C], 1)); df_ = D(tf.concat([fake, C], 1))
            ld = bce(tf.ones_like(dr), dr) + bce(tf.zeros_like(df_), df_)
            fm = tf.reduce_mean(tf.square(tf.reduce_mean(X, 0) - tf.reduce_mean(fake, 0)))
            lg = bce(tf.ones_like(df_), df_) + lam * fm
        og.apply_gradients(zip(tg.gradient(lg, G.trainable_variables), G.trainable_variables))
        od.apply_gradients(zip(td.gradient(ld, D.trainable_variables), D.trainable_variables))
        return ld, lg
    hist = []
    for e in range(epochs):
        a, b = step(); hist.append((float(a), float(b)))
    # class proportions identical to the original sample
    idx = np.random.RandomState(seed).choice(len(Z), n_out, replace=True)
    Cs = tf.one_hot(cls[idx], ncls)
    fake = G(tf.concat([tf.random.normal((n_out, noise)), Cs], 1)).numpy()
    return fake, hist


def synth_quality(real, syn, seed=0):
    fro = float(np.linalg.norm(np.corrcoef(real.T) - np.corrcoef(syn.T), "fro"))
    Xd = np.vstack([real, syn]); yd = np.r_[np.zeros(len(real)), np.ones(len(syn))]
    disc = float(cross_val_score(RandomForestClassifier(200, random_state=seed), Xd, yd, cv=5).mean())
    ks = int(sum(ks_2samp(real[:, j], syn[:, j]).pvalue > .05 for j in range(real.shape[1])))
    return dict(frobenius=fro, discriminative=disc, ks_ok=ks, ks_total=real.shape[1])


# ---------------------------------------------------------------- predictors (Table 7 of the article)
def make_model(name, seed, ysd=1.0):
    """The eight predictors with the best hyper-parameters retained in the thesis (Table 18)."""
    if name == "LR":
        m = LinearRegression(fit_intercept=True)
    elif name == "DT":
        m = DecisionTreeRegressor(max_depth=None, max_features="sqrt", min_samples_leaf=1,
                                  min_samples_split=2, random_state=seed)
    elif name == "RF":
        m = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_split=2, random_state=seed)
    elif name == "XGB":
        m = XGBRegressor(n_estimators=300, learning_rate=.1, max_depth=3, subsample=1.0,
                         colsample_bytree=.8, random_state=seed, verbosity=0, n_jobs=1)
    elif name == "MLP":
        m = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4,
                         learning_rate_init=.01, max_iter=1000, random_state=seed)
    elif name == "SVR":     # C=1000 and epsilon=1 are defined on TPF in million MAD (original unit): C*ysd, eps/ysd in standardised units
        m = SVR(kernel="linear", C=1000, epsilon=1.0 / ysd, gamma="scale")
    elif name == "GBR":
        m = GradientBoostingRegressor(n_estimators=100, learning_rate=.1, max_depth=5, subsample=1.0,
                                      random_state=seed)
    elif name == "ET":
        m = ExtraTreesRegressor(n_estimators=200, max_depth=5, min_samples_split=2, random_state=seed)
    return m


def fit_predict(name, Xtr, ytr, Xte, seed, ysd=1.0):
    m = make_model(name, seed, ysd).fit(Xtr, ytr)
    return m, m.predict(Xte)


def metrics(y, p, mu, sd):
    yr, pr = y * sd + mu, p * sd + mu           # back to million MAD for MAPE / RMSE_MAD
    return dict(MAE=mean_absolute_error(y, p), MSE=mean_squared_error(y, p),
                RMSE=np.sqrt(mean_squared_error(y, p)), R2=r2_score(y, p),
                MAPE=100 * np.mean(np.abs((yr - pr) / yr)),
                ExplainedVar=explained_variance_score(y, p), MedianAE=median_absolute_error(y, p),
                RMSE_MAD=np.sqrt(mean_squared_error(yr, pr)))


# ---------------------------------------------------------------- experiment
def run(scenario, seeds, featset="ALL17", log=print):
    df = load_real()
    feats = [c for c in df.columns if c not in ("date", TARGET) and c not in FEATSETS[featset]]
    cols = [TARGET] + feats
    res, preds, qual, hist, store, splits = [], {}, [], {}, {}, {}
    for s in seeds:
        t0 = time.time()
        tr, te = make_split(df, scenario, s, cols)
        mu = tr[cols].mean(); sd = tr[cols].std(ddof=0)
        Ztr = ((tr[cols] - mu) / sd).values; Zte = ((te[cols] - mu) / sd).values
        ymu, ysd = mu[TARGET], sd[TARGET]
        Xte, yte = Zte[:, 1:], Zte[:, 0]
        splits[s] = dict(tr=tr, te=te, Ztr=Ztr, Zte=Zte, ymu=ymu, ysd=ysd)
        # econometric benchmarks (same data, same scale): OLS and Poisson / PPML
        import statsmodels.api as sm
        try:
            yraw = Ztr[:, 0] * ysd + ymu
            poi = sm.GLM(yraw, sm.add_constant(Ztr[:, 1:]), family=sm.families.Poisson()).fit(maxiter=200)
            pr = poi.predict(sm.add_constant(Xte, has_constant="add"))
            p = (pr - ymu) / ysd
            res.append(dict(model="Poisson/PPML", gen="None", seed=s, **metrics(yte, p, ymu, ysd)))
            preds[("Poisson/PPML", "None", s)] = p
        except Exception as e:
            log("Poisson failed", e)
        syn = {}
        syn["VAE"], hist[("VAE", s)] = train_vae(Ztr, len(Ztr), s)
        syn["cGAN"], hist[("cGAN", s)] = train_cgan(Ztr, len(Ztr), 100 + s)
        for g in GENS:
            q = synth_quality(Ztr, syn[g], s); q.update(gen=g, seed=s); qual.append(q)
        store[s] = syn
        for m in MODELS:
            _, p = fit_predict(m, Ztr[:, 1:], Ztr[:, 0], Xte, s, ysd)       # reference: no augmentation
            res.append(dict(model=m, gen="None", seed=s, **metrics(yte, p, ymu, ysd))); preds[(m, "None", s)] = p
            for g in GENS:
                Za = np.vstack([Ztr, syn[g]])                              # (X*,Y*) = train U synthetic
                mod, p = fit_predict(m, Za[:, 1:], Za[:, 0], Xte, s, ysd)
                res.append(dict(model=m, gen=g, seed=s, **metrics(yte, p, ymu, ysd))); preds[(m, g, s)] = p
        log(f"[{scenario}] seed {s} done in {time.time()-t0:.0f}s")
    out = dict(scenario=scenario, featset=featset, results=pd.DataFrame(res), quality=pd.DataFrame(qual), preds=preds,
               feats=feats, splits=splits, syn=store, hist=hist)
    return out



# ================================================================== EXPLORATORY ANALYSIS
def eda():
    """Steps 1-5: descriptive statistics, ADF, IQR outliers, correlation/VIF/PCA,
    bivariate relations, variable selection and split figures. Output: OUT/eda_tables.xlsx + figures."""
    _style()
    df=load_real(); ALL=[c for c in df.columns if c!="date"]; EXP=[c for c in ALL if c!=TARGET]
    KEEP=[c for c in EXP if c not in FEATSETS["NOBAL"] and c!="PBC"]; DROP=[c for c in EXP if c not in KEEP]
    f2=lambda x,d=2: f"{x:,.{d}f}"
    # ---------- descriptive
    rows=[]
    for c in ALL:
        x=df[c]; d=0 if abs(x).max()>=1000 else (1 if abs(x).max()>=10 else 3)
        rows.append([c,f2(x.mean(),d),f2(x.median(),d),f2(x.std(),d),f2(x.min(),d),f2(x.quantile(.25),d),f2(x.quantile(.75),d),f2(x.max(),d),f2(x.std()/abs(x.mean())),f2(skew(x)),f2(kurtosis(x))])
    T2=rows
    fig,axs=plt.subplots(4,5,figsize=(15,10))
    for a,c in zip(axs.ravel(),ALL):
        sns.histplot(df[c],kde=True,ax=a,color=BLUE if c in KEEP+[TARGET] else GRY,edgecolor="white"); a.set_title(c,fontsize=10); a.set_xlabel(""); a.set_ylabel("")
    for a in axs.ravel()[len(ALL):]: a.axis("off")
    fig.suptitle(ar("توزيع المتغيرات الـ18 (أزرق: الهدف والمتغيرات المحتفظ بها، رمادي: المستبعدة لاحقاً)"),y=1.0); save("fig02_distributions")
    # ---------- ADF
    def adf(x): r=adfuller(pd.Series(x).dropna(),autolag="AIC"); return r[0],r[1]
    fp=lambda p: "<0.001" if p<0.001 else f"{p:.3f}"
    A=[]; adfd={}
    for c in ALL:
        x=df[c]; l=adf(x); d1=adf(x.diff()); 
        if (x>0).all(): ld=adf(np.log(x).diff()); lab="لوغاريتم + فرق"
        else: ld=d1; lab="فرق أول"
        ok=lambda p:p<.05
        if ok(l[1]): concl="مستقرة في المستوى"
        elif ok(d1[1]): concl="مستقرة بعد الفرق الأول"
        elif ok(ld[1]): concl="مستقرة بعد اللوغاريتم والفرق"
        else: concl="غير مستقرة"
        A.append([c,f"{l[0]:.3f}",fp(l[1]),f"{d1[0]:.3f}",fp(d1[1]),f"{ld[0]:.3f}",fp(ld[1]),concl]); adfd[c]=(l[1],d1[1],ld[1])
    fig,ax=plt.subplots(figsize=(11,4.2)); w=.27; xx=np.arange(len(ALL))
    ax.bar(xx-w,[adfd[c][0] for c in ALL],w,color=RED,label=ar("المستوى")); ax.bar(xx,[adfd[c][1] for c in ALL],w,color=ORG,label=ar("الفرق الأول")); ax.bar(xx+w,[adfd[c][2] for c in ALL],w,color=GRN,label=ar("لوغاريتم+فرق أو فرق"))
    ax.axhline(.05,color="k",ls="--",lw=1); ax.set_xticks(xx); ax.set_xticklabels(ALL,rotation=40,ha="right",fontsize=8); ax.set_ylabel("p-value"); ax.legend(fontsize=8)
    ax.set_title(ar("القيمة الاحتمالية لاختبار ADF لكل متغير حسب المرحلة")); save("fig04_ADF_pvalues")
    # ---------- IQR
    clean,mask=iqr_filter(df,ALL); q1,q3=df[ALL].quantile(.25),df[ALL].quantile(.75); iq=q3-q1
    I=[]
    for c in ALL:
        d=0 if abs(df[c]).max()>=1000 else (1 if abs(df[c]).max()>=10 else 3)
        I.append([c,f2(q1[c],d),f2(q3[c],d),f2(iq[c],d),f2(q1[c]-1.5*iq[c],d),f2(q3[c]+1.5*iq[c],d),str(int(mask[c].sum())),f"{100*mask[c].mean():.1f}"])
    print("rows removed (all 19):",len(df)-len(clean),"after",len(clean))
    clean8,mask8=iqr_filter(df,[TARGET]+KEEP); print("rows removed (8 vars):",len(df)-len(clean8))
    sel=["NRR","DIR","CF","FA","HJ","GR %","IR (%)",TARGET]
    fig,ax=plt.subplots(2,4,figsize=(13,6))
    for j,c in enumerate(sel):
        a=ax.ravel()[j]; a.boxplot([df[c],clean[c]],tick_labels=[ar("قبل"),ar("بعد")],patch_artist=True,boxprops=dict(facecolor="#bee3f8"),medianprops=dict(color=RED)); a.set_title(c,fontsize=10)
    fig.suptitle(ar("صناديق المتغيرات قبل حذف القيم الشاذة وبعده (IQR)"),y=1.01); save("fig05_boxplots_IQR")
    flag=mask.any(axis=1).values
    fig,ax=plt.subplots(figsize=(11,2.2)); ax.bar(df.date,np.ones(len(df)),width=26,color=[RED if f else "#bee3f8" for f in flag]); ax.set_yticks([])
    ax.set_title(ar(f"الأشهر المحذوفة بقاعدة IQR (بالأحمر): {len(df)-len(clean)} شهراً من {len(df)}"))
    for sp in ax.spines.values(): sp.set_visible(False)
    save("fig06_removed_months")
    # ---------- correlation, VIF, PCA
    cm=df[ALL].corr(); ct=cm[TARGET]
    Z=(df[EXP]-df[EXP].mean())/df[EXP].std(); v=pd.Series([vif(Z.values,i) for i in range(Z.shape[1])],index=EXP)
    Z8=(df[KEEP]-df[KEEP].mean())/df[KEEP].std(); v8=pd.Series([vif(Z8.values,i) for i in range(Z8.shape[1])],index=KEEP)
    plt.figure(figsize=(11,9)); sns.heatmap(cm,annot=True,fmt=".2f",cmap="coolwarm",vmin=-1,vmax=1,annot_kws={"size":6}); plt.title(ar("مصفوفة الارتباط لكل المتغيرات")); save("fig07_correlation")
    fig,ax=plt.subplots(1,2,figsize=(12,5),gridspec_kw={"width_ratios":[1.4,1]})
    vs=v.sort_values(); ax[0].barh(vs.index,vs.values,color=[BLUE if i in KEEP else GRY for i in vs.index]); ax[0].set_xscale("log"); ax[0].axvline(10,color=RED,ls="--")
    ax[0].set_title(ar("VIF للمتغيرات التفسيرية الـ17 (أزرق = المحتفظ به)")); ax[0].set_xlabel("VIF (log)")
    v8s=v8.sort_values(); ax[1].barh(v8s.index,v8s.values,color=BLUE); ax[1].axvline(10,color=RED,ls="--"); ax[1].set_title(ar("VIF بعد الانتقاء (8 متغيرات)")); ax[1].set_xlabel("VIF"); save("fig08_VIF")
    pa=PCA().fit(Z); eva=pa.explained_variance_ratio_*100; sca=pa.transform(Z)
    p8=PCA().fit(Z8); ev8=p8.explained_variance_ratio_*100
    fig,ax=plt.subplots(1,2,figsize=(12.5,4.8))
    ax[0].bar(range(1,len(eva)+1),eva,color=BLUE); ax[0].plot(range(1,len(eva)+1),np.cumsum(eva),"o-",color=ORG); ax[0].set_xlabel("PC"); ax[0].set_ylabel("%"); ax[0].set_title(ar("التباين المفسَّر والتراكمي (17 متغيراً)"))
    pp=ax[1].scatter(sca[:,0],sca[:,1],c=df[TARGET],cmap="coolwarm",s=30,edgecolor="k",lw=.3)
    for i,c in enumerate(EXP):
        ax[1].arrow(0,0,pa.components_[0,i]*5,pa.components_[1,i]*5,color=GRY,head_width=.08); ax[1].text(pa.components_[0,i]*5.4,pa.components_[1,i]*5.4,c,fontsize=7)
    ax[1].set_xlabel(f"PC1 ({eva[0]:.1f}%)"); ax[1].set_ylabel(f"PC2 ({eva[1]:.1f}%)"); ax[1].set_title(ar("إسقاط PCA ملوَّن بقيمة TPF")); plt.colorbar(pp,ax=ax[1],label="TPF"); save("fig09_PCA")
    # decision table
    reason={"REF":"مكوّن من مكونات TPF","CF":"مكوّن من مكونات TPF","EF":"مكوّن من مكونات TPF","TF":"مكوّن من مكونات TPF","TA":"رصيد ميزانية يحتوي الهدف","RC":"رصيد ميزانية يتطابق تقريباً مع TA","CCA":"رصيد ميزانية ينمو مع الهدف","ID":"رصيد ميزانية ينمو مع الهدف","TL":"رصيد ميزانية","PBC":"يطابق الهدف تقريباً (تسرب بيانات)"}
    order=KEEP+DROP; S=[]
    for c in order:
        S.append([c,f"{ct[c]:.3f}",f"{v[c]:,.1f}",f"{v8[c]:.2f}" if c in KEEP else "-","احتفاظ" if c in KEEP else "حذف",reason.get(c,"مستقل نسبياً عن الهدف، VIF منخفض")])
    pc_rows=[[f"PC{i+1}",f"{eva[i]:.2f}",f"{np.cumsum(eva)[i]:.2f}",f"{ev8[i]:.2f}" if i<len(ev8) else "-",f"{np.cumsum(ev8)[i]:.2f}" if i<len(ev8) else "-"] for i in range(8)]

    V9=[TARGET]+KEEP
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(df.date, df[TARGET], color=BLUE, lw=2)
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), color=RED, alpha=.12)
    ax.axvspan(pd.Timestamp("2022-07-01"), pd.Timestamp("2023-03-31"), color=ORG, alpha=.15)
    ax.text(pd.Timestamp("2020-02-01"), 44000, ar("جائحة كوفيد-19"), color=RED, fontsize=9)
    ax.text(pd.Timestamp("2022-07-15"), 40000, ar("ذروة التضخم"), color=ORG, fontsize=9)
    ax.set_title(ar("تطور إجمالي التمويل التشاركي (مليون درهم)، أكتوبر 2018 – يونيو 2026"))
    ax.set_ylabel("TPF"); save("fig01_serie_TPF")
    t = df[TARGET]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(df.date, t, color=BLUE); ax[0].set_title(ar("المستوى الأصلي"))
    ax[1].plot(df.date, np.log(t), color=ORG); ax[1].set_title(ar("اللوغاريتم"))
    ax[2].plot(df.date[1:], np.log(t).diff().dropna(), color=GRN); ax[2].axhline(0, color=GRY, lw=.8)
    ax[2].set_title(ar("اللوغاريتم + الفرق الأول"))
    fig.suptitle(ar("استقرارية سلسلة TPF عبر ثلاث مراحل"), y=1.04); save("fig03_ADF_TPF")
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
    tr, te = make_split(df, "IQR_random", 0, V9)
    ax[0].scatter(df.date, df[TARGET], c=GRY, s=14, label=ar("محذوفة (شاذة)"))
    ax[0].scatter(tr.date, tr[TARGET], c=BLUE, s=22, label=ar("تدريب") + f" ({len(tr)})")
    ax[0].scatter(te.date, te[TARGET], c=ORG, s=30, marker="s", label=ar("اختبار") + f" ({len(te)})")
    ax[0].set_title(ar("التقسيم العشوائي 80/20 بعد حذف IQR (المواصفة الرئيسية)")); ax[0].legend(fontsize=8)
    trc, tec = make_split(df, "FULL_chrono", 0, V9)
    ax[1].scatter(trc.date, trc[TARGET], c=BLUE, s=22, label=ar("تدريب") + f" ({len(trc)})")
    ax[1].scatter(tec.date, tec[TARGET], c=ORG, s=30, marker="s", label=ar("اختبار") + f" ({len(tec)})")
    ax[1].set_title(ar("التقسيم الزمني بلا حذف (اختبار الحساسية)")); ax[1].legend(fontsize=8); save("fig10_split")
    # ---------- bivariate relations (Pearson, Spearman, linear and quadratic R2)
    y=df[TARGET]; rows=[]
    for c in EXP:
        x=df[c]; r,p=stats.pearsonr(x,y); rho,ps=stats.spearmanr(x,y)
        sl,ic,_,_,_=stats.linregress(x,y)
        # quadratic gain
        X2=np.column_stack([np.ones(len(x)),(x-x.mean())/x.std(),((x-x.mean())/x.std())**2]); b=np.linalg.lstsq(X2,y,rcond=None)[0]; r2q=1-((y-X2@b)**2).sum()/((y-y.mean())**2).sum()
        rows.append(dict(var=c,r=r,p=p,rho=rho,ps=ps,R2=r*r,R2q=r2q))
    T=pd.DataFrame(rows).set_index("var"); T=T.reindex(T.r.abs().sort_values(ascending=False).index)
    print(T.round(4).to_string())
    fig,axs=plt.subplots(5,4,figsize=(15,19)); order=list(T.index)
    yr=df.date.dt.year+df.date.dt.month/12
    for a,c in zip(axs.ravel(),order):
        sc=a.scatter(df[c],y,c=yr,cmap="viridis",s=16)
        sl,ic,_,_,_=stats.linregress(df[c],y); xs=np.linspace(df[c].min(),df[c].max(),50); a.plot(xs,ic+sl*xs,color="#c53030",lw=1.3)
        nm={"GR %":"GR","IR (%)":"IR"}.get(c,c)
        a.set_title(f"{nm}   r={T.loc[c,'r']:.2f}  ρ={T.loc[c,'rho']:.2f}",fontsize=11,color="#2b6cb0" if c in KEEP else "#4a5568")
        a.set_xlabel(nm,fontsize=8); a.set_ylabel("TPF",fontsize=8)
    for a in axs.ravel()[len(order):]: a.axis("off")
    fig.colorbar(sc,ax=axs.ravel().tolist(),shrink=.4,label=ar("السنة"),location="bottom",pad=.03)
    fig.suptitle(ar("العلاقة الثنائية بين كل متغير تفسيري والمتغير الهدف TPF (أزرق: المحتفظ بها، رمادي: المستبعدة لاحقاً)"),y=.985,fontsize=13)
    fig.subplots_adjust(hspace=.42,wspace=.28,top=.955,bottom=.07)
    plt.savefig(f"{FIG}/fig26_bivariate_scatter.png",dpi=150,bbox_inches="tight"); plt.close()
    BIV=T.copy()

    # ---------- export all exploratory tables
    with pd.ExcelWriter(os.path.join(OUT, "eda_tables.xlsx")) as w:
        pd.DataFrame(T2,columns=["variable","mean","median","std","min","Q1","Q3","max","CV","skew","kurtosis"]).to_excel(w,sheet_name="01_descriptive",index=False)
        pd.DataFrame(A,columns=["variable","ADF_level","p_level","ADF_diff","p_diff","ADF_logdiff_or_diff","p","conclusion"]).to_excel(w,sheet_name="02_ADF",index=False)
        pd.DataFrame(I,columns=["variable","Q1","Q3","IQR","lower","upper","n_outliers","pct"]).to_excel(w,sheet_name="03_IQR",index=False)
        pd.DataFrame(S,columns=["variable","corr_with_TPF","VIF_17","VIF_8","decision","reason"]).to_excel(w,sheet_name="04_selection",index=False)
        pd.DataFrame(pc_rows,columns=["PC","var_17_%","cum_17_%","var_8_%","cum_8_%"]).to_excel(w,sheet_name="04_PCA",index=False)
        BIV.round(4).to_excel(w,sheet_name="05_bivariate")
        pd.DataFrame({"removed_months":[d.strftime("%Y-%m") for d in df.date[flag]]}).to_excel(w,sheet_name="03_removed_months",index=False)
    print("rows removed:", len(df)-len(clean), "-> remaining", len(clean))



# ================================================================== RESULTS, TESTS, FIGURES
def analyze():
    """Steps 6-11: tables of the eight models, augmentation effect (Wilcoxon), Diebold-Mariano,
    Friedman, synthetic-data quality, learning curve, SHAP, comparison with traditional models,
    sensitivity analysis. Needs the four runs (run_all) in WORK. Output: OUT/results_tables.xlsx + figures."""
    _style()
    import shap
    from matplotlib.patches import FancyBboxPatch
    COL = {"None": GRY, "VAE": BLUE, "cGAN": ORG}
    DLAB = {"None": ar("حقيقي فقط"), "VAE": "VAE", "cGAN": "cGAN"}
    P, FR, FC, A17 = L("IQR_random", "NOBAL"), L("FULL_random", "NOBAL"), L("FULL_chrono", "NOBAL"), L("IQR_random", "ALL17")
    N = {}
    NS = len(P["splits"]); SEEDS = range(NS); GENS3 = ["None", "VAE", "cGAN"]
    MET = ["MAE", "MSE", "RMSE", "R2", "MAPE", "ExplainedVar", "MedianAE", "RMSE_MAD"]

    def agg(r):
        g = r["results"].groupby(["model", "gen"])[MET]
        return g.mean(), g.std()
    m, s = agg(P)
    T = m.copy(); T["R2_sd"] = s["R2"]; T["RMSE_sd"] = s["RMSE"]
    N["table21"] = T.loc[[(a, b) for a in MODELS for b in GENS3]].round(4).reset_index().to_dict("records")
    N["poisson"] = T.loc[("Poisson/PPML", "None")].round(4).to_dict()
    hyb = T.loc[[(a, b) for a in MODELS for b in GENS]]; allc = T.loc[[(a, b) for a in MODELS for b in GENS3]]
    best_h = hyb["R2"].idxmax(); best_all = allc["R2"].idxmax()
    N["best_hybrid"] = list(best_h); N["best_all"] = list(best_all)
    N["rank_hybrids"] = hyb.sort_values("R2", ascending=False).round(4).reset_index().to_dict("records")

    # ---------- augmentation effect (paired over repetitions)
    rr = P["results"]; aug = []
    for mo in MODELS:
        base = rr[(rr.model == mo) & (rr.gen == "None")].sort_values("seed").RMSE.values
        for g in GENS:
            a = rr[(rr.model == mo) & (rr.gen == g)].sort_values("seed").RMSE.values
            try: pw = stats.wilcoxon(a, base).pvalue
            except Exception: pw = np.nan
            aug.append(dict(model=mo, gen=g, rmse_none=base.mean(), rmse_aug=a.mean(), change_pct=100 * (a.mean() / base.mean() - 1),
                            improved=int((a < base).sum()), n=len(a), p_wilcoxon=pw))
    aug = pd.DataFrame(aug); N["aug"] = aug.round(4).to_dict("records")

    # ---------- synthetic quality
    q = P["quality"].groupby("gen")[["frobenius", "discriminative", "ks_ok", "ks_total"]].agg(["mean", "std"]); N["quality"] = {g: {f"{a}_{b}": float(q.loc[g, (a, b)]) for a in ["frobenius", "discriminative", "ks_ok", "ks_total"] for b in ["mean", "std"]} for g in GENS}

    # ---------- DM
    def dm(e1, e2):
        d = e1 ** 2 - e2 ** 2; T_ = len(d); v = d.var(ddof=1) / T_
        if v == 0: return np.nan, np.nan
        st = d.mean() / np.sqrt(v) * np.sqrt((T_ - 1) / T_); return st, 2 * (1 - stats.t.cdf(abs(st), T_ - 1))
    def dm_vs(ref, opp):
        out = []
        for sd_ in SEEDS:
            y = P["splits"][sd_]["Zte"][:, 0]
            st, p = dm(y - P["preds"][(ref[0], ref[1], sd_)], y - P["preds"][(opp[0], opp[1], sd_)]); out.append((st, p))
        a = np.array(out); return float(np.nanmean(a[:, 0])), float(np.nanmedian(a[:, 1])), int((a[:, 1] < .05).sum())
    dmr = []
    for opp in [(mo, "None") for mo in MODELS if mo != best_h[0]] + [("Poisson/PPML", "None")]:
        if opp == best_h: continue
        a, b, c = dm_vs(best_h, opp); dmr.append(dict(opponent=f"{opp[0]}", DM_mean=a, p_median=b, seeds_sig=c, n=NS))
    dmr = pd.DataFrame(dmr); N["dm"] = dmr.round(4).to_dict("records")
    # the best real-only model vs LR and Poisson
    dmb = []
    for opp in [("LR", "None"), ("Poisson/PPML", "None")]:
        a, b, c = dm_vs(best_all, opp); dmb.append(dict(opponent=opp[0], DM_mean=a, p_median=b, seeds_sig=c))
    N["dm_best_all"] = pd.DataFrame(dmb).round(4).to_dict("records")

    # ---------- sensitivity
    sens = {}
    for lab, r in [("IQR_random", P), ("FULL_random", FR), ("FULL_chrono", FC), ("ALL17", A17)]:
        mm, _ = agg(r); d = {}
        for mo in MODELS:
            d[mo] = dict(real=float(mm.loc[(mo, "None"), "R2"]), hyb=float(np.mean([mm.loc[(mo, g), "R2"] for g in GENS])),
                         rmse_real=float(mm.loc[(mo, "None"), "RMSE"]))
        d["Poisson"] = dict(real=float(mm.loc[("Poisson/PPML", "None"), "R2"]), hyb=np.nan, rmse_real=float(mm.loc[("Poisson/PPML", "None"), "RMSE"]))
        sens[lab] = d
    N["sens"] = sens

    # ====================== FIGURES ======================
    # fig11: workflow
    fig, ax = plt.subplots(figsize=(11, 4.4)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 4.4)
    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=c, ec="#2d3748", lw=1)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)
    box(.1, 1.7, 1.9, 1.0, ar("البيانات الشهرية\n93 مشاهدة × 9 متغيرات"), "#bee3f8")
    box(2.4, 1.7, 2.0, 1.0, ar("المعالجة: ADF و IQR\nتوحيد قياسي، تقسيم 80/20"), "#c6f6d5")
    box(4.9, 3.0, 2.0, .9, ar("VAE (تدريب فقط)"), "#fefcbf"); box(4.9, .5, 2.0, .9, ar("cGAN (تدريب فقط)"), "#fefcbf")
    box(7.4, 1.7, 1.8, 1.0, ar("حقيقي + مولَّد\n(مجموعة التدريب)"), "#fed7d7")
    box(9.5, 1.7, 1.4, 1.0, ar("8 نماذج\nتنبؤية"), "#e9d8fd")
    for x1, y1, x2, y2 in [(2.0, 2.2, 2.4, 2.2), (4.4, 2.35, 4.9, 3.35), (4.4, 2.05, 4.9, .95), (6.9, 3.35, 7.4, 2.5), (6.9, .95, 7.4, 1.9), (9.2, 2.2, 9.5, 2.2)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.text(5.5, .05, ar("التقييم على بيانات الاختبار الحقيقية فقط: 7 مقاييس + تشخيص البواقي + SHAP + مقارنة تقليدية + DM"), ha="center", fontsize=9)
    ax.set_title(ar("مخطط المنهجية الهجينة (تنبؤية + توليدية)")); save("fig11_workflow")

    # fig12: training curves of generators (seed 0)
    hv = np.array(P["hist"][("VAE", 0)]); hg = np.array(P["hist"][("cGAN", 0)])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(hv[:, 0], color=BLUE); ax[0].set_yscale("log"); ax[0].set_title(ar("VAE: خطأ إعادة البناء")); ax[0].set_xlabel("epoch")
    ax[1].plot(hv[:, 1], color=ORG); ax[1].set_title(ar("VAE: تباعد KL")); ax[1].set_xlabel("epoch")
    ax[2].plot(hg[:, 0], color=BLUE, label="D", alpha=.8); ax[2].plot(hg[:, 1], color=ORG, label="G", alpha=.8)
    ax[2].set_title(ar("cGAN: خسارة المميّز والمولّد")); ax[2].legend(); ax[2].set_xlabel("epoch"); save("fig12_generator_losses")

    # fig13: real vs synthetic
    S0 = P["splits"][0]; Zt = S0["Ztr"]; pc = PCA(2).fit(Zt)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].scatter(*pc.transform(Zt).T, c="k", s=22, label=ar("حقيقي"))
    for g in GENS: ax[0].scatter(*pc.transform(P["syn"][0][g]).T, c=COL[g], s=22, alpha=.65, label=g)
    ax[0].set_title(ar("إسقاط PCA: بيانات حقيقية ومولَّدة")); ax[0].legend()
    ax[1].hist(Zt[:, 0], bins=10, density=True, alpha=.5, color="k", label=ar("حقيقي"))
    for g in GENS: ax[1].hist(P["syn"][0][g][:, 0], bins=10, density=True, alpha=.5, color=COL[g], label=g)
    ax[1].set_title(ar("توزيع TPF (معياري)")); ax[1].legend(); save("fig13_real_vs_synthetic")

    # fig14: quality metrics
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, (k, t_, good) in zip(ax, [("frobenius", "فرق مصفوفة الارتباط (الأقل أفضل)", None), ("discriminative", "دقة التمييز (الأقرب لـ 0.5 أفضل)", .5), ("ks_ok", "متغيرات مطابقة بـ KS (من 9)", None)]):
        vals = [N["quality"][g][f"{k}_mean"] for g in GENS]; err = [N["quality"][g][f"{k}_std"] for g in GENS]
        a.bar(GENS, vals, yerr=err, color=[COL[g] for g in GENS], capsize=4); a.set_title(ar(t_), fontsize=10)
        if good: a.axhline(good, color=RED, ls="--")
    save("fig14_synthetic_quality")

    # fig15: grouped bars R2
    fig, ax = plt.subplots(figsize=(10, 5)); w = .26; xx = np.arange(len(MODELS))
    for i, g in enumerate(GENS3):
        ax.bar(xx + (i - 1) * w, [T.loc[(mo, g), "R2"] for mo in MODELS], w, yerr=[T.loc[(mo, g), "R2_sd"] for mo in MODELS],
               color=COL[g], label=DLAB[g], capsize=2, error_kw=dict(lw=.8))
    ax.set_xticks(xx); ax.set_xticklabels(MODELS); ax.set_ylim(.3, 1.12); ax.set_ylabel("R²")
    ax.legend(ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.08), frameon=False)
    ax.set_title(ar("معامل التحديد R² للنماذج الثمانية حسب بيانات التدريب") + "\n" + ar("(متوسط 10 تكرارات ± الانحراف المعياري)")); save("fig15_R2_by_model")

    # fig16: heatmaps
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, k, cm_, t_ in [(ax[0], "R2", "YlGn", "R²"), (ax[1], "RMSE", "YlOrRd", "RMSE")]:
        hm = pd.DataFrame({DLAB[g] if g != "None" else "Real": [T.loc[(mo, g), k] for mo in MODELS] for g in GENS3}, index=MODELS)
        sns.heatmap(hm, annot=True, fmt=".3f", cmap=cm_, ax=a, cbar=False); a.set_title(t_)
    save("fig16_heatmaps")

    # fig17: augmentation effect
    fig, ax = plt.subplots(figsize=(10, 4)); w = .38; xx = np.arange(len(MODELS))
    for i, g in enumerate(GENS):
        sub = aug[aug.gen == g].set_index("model").loc[MODELS]
        ax.bar(xx + (i - .5) * w, sub.change_pct, w, color=COL[g], label=g)
        for xi, (v, k) in enumerate(zip(sub.change_pct, sub.improved)):
            ax.text(xi + (i - .5) * w, v + (2 if v >= 0 else -6), f"{k}/{NS}", ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(xx); ax.set_xticklabels(MODELS); ax.set_ylabel(ar("تغير RMSE (%)")); ax.legend()
    ax.set_title(ar("أثر البيانات المولَّدة على RMSE (القيم السالبة = تحسّن؛ الأرقام = عدد مرات التحسن)")); save("fig17_augmentation_effect")

    # fig18-21 diagnostics (8 panels)
    def panels(fn, drawer, title, h=7.2):
        fig, axs = plt.subplots(2, 4, figsize=(15, h))
        for a, mo in zip(axs.ravel(), MODELS): drawer(a, mo); a.set_title(mo, fontsize=11)
        h_, l_ = axs.ravel()[0].get_legend_handles_labels()
        fig.suptitle(ar(title), y=1.0, fontsize=12); save(fn)
    def d_scatter(a, mo):
        y = S0["Zte"][:, 0]
        for g in GENS3: a.scatter(y, P["preds"][(mo, g, 0)], s=22, c=COL[g], alpha=.75, label=DLAB[g])
        lim = [y.min() - .2, y.max() + .2]; a.plot(lim, lim, "k--", lw=1); a.set_xlabel(ar("الفعلي")); a.set_ylabel(ar("المتوقع"))
        if mo == "LR": a.legend(fontsize=7)
    panels("fig18_actual_vs_predicted", d_scatter, "القيم الفعلية مقابل المتوقعة (التقسيم الأول، 11 مشاهدة اختبار)")
    def pooled(mo, g):
        return (np.concatenate([P["splits"][k]["Zte"][:, 0] - P["preds"][(mo, g, k)] for k in SEEDS]),
                np.concatenate([P["preds"][(mo, g, k)] for k in SEEDS]))
    def d_hist(a, mo):
        for g in GENS3: a.hist(pooled(mo, g)[0], bins=18, histtype="step", lw=1.6, color=COL[g], label=DLAB[g])
        a.axvline(0, color="k", lw=.8); a.set_xlabel(ar("الخطأ"))
        if mo == "LR": a.legend(fontsize=7)
    panels("fig19_error_histograms", d_hist, "توزيع أخطاء التنبؤ (تجميع 10 تكرارات)")
    def d_res(a, mo):
        for g in GENS3:
            e, p_ = pooled(mo, g); a.scatter(p_, e, s=10, c=COL[g], alpha=.5, label=DLAB[g])
        a.axhline(0, color="k", ls="--", lw=.8); a.set_xlabel(ar("المتوقع")); a.set_ylabel(ar("البواقي"))
        if mo == "LR": a.legend(fontsize=7)
    panels("fig20_residuals", d_res, "البواقي مقابل القيم المتوقعة (تجميع 10 تكرارات)")
    rows = []
    for mo in MODELS:
        for g in GENS3: rows += [dict(model=mo, data=DLAB[g], abs_err=abs(v)) for v in pooled(mo, g)[0]]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    sns.boxplot(data=pd.DataFrame(rows), x="model", y="abs_err", hue="data", palette=[GRY, BLUE, ORG], ax=ax, fliersize=2, linewidth=.9)
    ax.set_ylabel(ar("الخطأ المطلق")); ax.set_xlabel(""); ax.legend(fontsize=8, ncol=3); ax.set_title(ar("صناديق الخطأ المطلق للنماذج الثمانية")); save("fig21_abs_error_boxplot")
    absr = pd.DataFrame(rows).groupby(["model", "data"]).abs_err.agg(["median", "max"]).round(3).reset_index(); N["abs_err_summary"] = absr.to_dict("records")

    # fig22 learning curve, fig23 SHAP (best hybrid)
    mo, g = best_h; ysd = S0["ysd"]
    Za = np.vstack([S0["Ztr"], P["syn"][0][g]])
    sz, trs, vas = learning_curve(make_model(mo, 0, ysd), Za[:, 1:], Za[:, 0], cv=KFold(5, shuffle=True, random_state=0), scoring="r2", train_sizes=np.linspace(.2, 1, 8))
    plt.figure(figsize=(7, 4.4))
    for arr, c, l in [(trs, BLUE, ar("R² التدريب")), (vas, RED, ar("R² التحقق"))]:
        plt.plot(sz, arr.mean(1), "-o", color=c, label=l); plt.fill_between(sz, arr.mean(1) - arr.std(1), arr.mean(1) + arr.std(1), color=c, alpha=.15)
    plt.xlabel(ar("عدد مشاهدات التدريب")); plt.ylabel("R²"); plt.legend(); plt.title(ar(f"منحنى التعلم: {mo} + {g}")); save("fig22_learning_curve")
    N["learning_curve"] = dict(size=[int(x) for x in sz], train=[float(x) for x in trs.mean(1)], val=[float(x) for x in vas.mean(1)])
    import shap
    mod = make_model(mo, 0, ysd).fit(Za[:, 1:], Za[:, 0])
    Xs = np.vstack([S0["Ztr"][:, 1:], S0["Zte"][:, 1:]])
    try: sv = shap.TreeExplainer(mod).shap_values(Xs)
    except Exception: sv = shap.Explainer(mod.predict, Za[:, 1:])(Xs).values
    imp = pd.Series(np.abs(sv).mean(0), index=P["feats"]).sort_values(ascending=False); N["shap"] = imp.round(4).to_dict()
    plt.figure(figsize=(7, 4)); imp.sort_values().plot.barh(color=BLUE); plt.xlabel("mean(|SHAP|)"); plt.title(ar(f"أهمية المتغيرات (SHAP): {mo} + {g}")); save("fig23a_shap_importance")
    shap.summary_plot(sv, Xs, feature_names=P["feats"], show=False); plt.title(ar("اتجاه تأثير المتغيرات (SHAP)")); save("fig23b_shap_beeswarm")

    # fig24 comparison with traditional models; fig25 sensitivity
    fig, ax = plt.subplots(figsize=(10, 4.4))
    names = MODELS + ["Poisson/PPML"]; vals = [T.loc[(mo_, "None"), "RMSE_MAD"] for mo_ in MODELS] + [T.loc[("Poisson/PPML", "None"), "RMSE_MAD"]]
    cols_ = [RED if n_ in ("LR", "Poisson/PPML") else BLUE for n_ in names]
    ax.bar(names, vals, color=cols_); ax.set_ylabel(ar("RMSE (مليون درهم)")); ax.set_title(ar("خطأ التنبؤ للنماذج مقارنة بالنماذج التقليدية (أحمر) - بيانات حقيقية فقط"))
    for i, v in enumerate(vals): ax.text(i, v + 120, f"{v:,.0f}", ha="center", fontsize=8)
    save("fig24_vs_traditional")
    fig, ax = plt.subplots(figsize=(11, 4.4)); w = .27; xx = np.arange(len(MODELS))
    for i, (lab, c, t_) in enumerate([("IQR_random", BLUE, "IQR + عشوائي (رئيسي)"), ("FULL_random", GRN, "بلا حذف + عشوائي"), ("ALL17", RED, "17 متغيراً (تسرب)")]):
        ax.bar(xx + (i - 1) * w, [sens[lab][mo_]["real"] for mo_ in MODELS], w, color=c, label=ar(t_))
    ax.set_xticks(xx); ax.set_xticklabels(MODELS); ax.set_ylim(.6, 1.02); ax.set_ylabel("R² (real only)"); ax.legend(fontsize=8, ncol=3)
    ax.set_title(ar("اختبارات الحساسية: R² بيانات حقيقية فقط")); save("fig25_sensitivity")

    # ====================== EXCEL ======================
    def fmt(r):
        a, b = agg(r); return a.round(4).join(b.add_suffix("_sd").round(4)).reset_index()
    with pd.ExcelWriter(os.path.join(OUT, "results_tables.xlsx")) as w:
        fmt(P).to_excel(w, sheet_name="08_models_primary", index=False)
        aug.round(4).to_excel(w, sheet_name="08_augmentation_effect", index=False)
        pd.DataFrame(N["quality"]).T.round(4).to_excel(w, sheet_name="06_synthetic_quality")
        dmr.round(4).to_excel(w, sheet_name="10_DM_best_hybrid", index=False)
        imp.rename("mean_abs_SHAP").round(4).to_excel(w, sheet_name="09_SHAP")
        fmt(FR).to_excel(w, sheet_name="10_sens_no_removal", index=False); fmt(FC).to_excel(w, sheet_name="10_sens_chrono", index=False); fmt(A17).to_excel(w, sheet_name="10_sens_17vars", index=False)
    json.dump(N, open(os.path.join(OUT, "numbers_results.json"), "w"), ensure_ascii=False, default=float, indent=1)
    pd.set_option("display.width", 220)
    print("BEST hybrid", best_h, "BEST all", best_all)
    print(T.loc[[(a, b) for a in MODELS for b in GENS3], ["MAE", "RMSE", "R2", "R2_sd", "MAPE", "ExplainedVar", "MedianAE", "RMSE_MAD"]].round(4).to_string())
    print(T.loc[("Poisson/PPML", "None")].round(4).to_dict())
    print(aug.round(3).to_string()); print(dmr.round(3).to_string()); print(N["dm_best_all"]); print(N["quality"]); print(imp.round(3).to_dict())
    for k, v in sens.items(): print(k, {a: round(b["real"], 3) for a, b in v.items()})
    print(N["learning_curve"])


    # ---------- Friedman test (differences between the eight models) and stability of the best model
    from collections import Counter
    fr_out = {}
    for lab, r in [("IQR_random", P), ("FULL_random", FR)]:
        rs = r["results"]
        M = np.array([rs[(rs.model == m_) & (rs.gen == "None")].sort_values("seed").RMSE.values for m_ in MODELS]).T
        chi, p_ = stats.friedmanchisquare(*M.T)
        ranks = stats.rankdata(M, axis=1).mean(0)
        best = Counter(MODELS[i] for i in M.argmin(1))
        fr_out[lab] = dict(chi2=float(chi), p=float(p_), mean_ranks={m_: round(float(x), 2) for m_, x in zip(MODELS, ranks)}, best_counts=dict(best))
        print("Friedman", lab, fr_out[lab])
    N["friedman"] = fr_out
    json.dump(N, open(os.path.join(OUT, "numbers_results.json"), "w"), ensure_ascii=False, default=float, indent=1)



# ================================================================== HYPOTHESIS TESTS
def hypotheses():
    """Hypothesis tests H1-H3: OLS (levels and growth rates, HAC errors), incremental F-test,
    and predictive comparison bank-only vs bank+macro (paired Wilcoxon)."""
    import statsmodels.api as sm
    df=load_real(); KEEP=[c for c in df.columns if c not in("date",TARGET) and c not in FEATSETS["NOBAL"]]
    BANK=["NRR","DIR","SC","CAR","FA","HJ"]; MAC=["GR %","IR (%)"]; out={}
    # ---- H1/H2 explanatory tests, levels (93 obs) with HAC SE
    def ols(y,X,lag=3):
        X=sm.add_constant(X); m=sm.OLS(y,X).fit(cov_type="HAC",cov_kwds={"maxlags":lag}); return m
    y=df[TARGET]
    m1=ols(y,df[BANK]); m2=ols(y,df[BANK+MAC])
    full=sm.OLS(y,sm.add_constant(df[BANK+MAC])).fit(); red=sm.OLS(y,sm.add_constant(df[BANK])).fit()
    F,pF,_=full.compare_f_test(red)
    print("LEVELS bank only R2",round(m1.rsquared,4),"F-p",m1.f_pvalue)
    print(m1.pvalues.round(4).to_dict())
    print("bank+macro R2",round(m2.rsquared,4)); print(m2.pvalues.round(4).to_dict())
    print("incremental F (add GR,IR) F=",round(F,3),"p=",round(pF,4))
    # ---- differences (stationary) : dlogTPF on d(vars), GR level
    dd=pd.DataFrame({"dlogTPF":np.log(df[TARGET]).diff()}); 
    for c in BANK+["IR (%)"]: dd[c]=df[c].diff()
    dd["GR %"]=df["GR %"]; dd=dd.dropna()
    d1=ols(dd["dlogTPF"],dd[BANK]); d2=ols(dd["dlogTPF"],dd[BANK+MAC])
    f2=sm.OLS(dd["dlogTPF"],sm.add_constant(dd[BANK+MAC])).fit(); r2=sm.OLS(dd["dlogTPF"],sm.add_constant(dd[BANK])).fit()
    Fd,pFd,_=f2.compare_f_test(r2)
    print("DIFFS bank only R2",round(d1.rsquared,4),"F-p",round(d1.f_pvalue,4)); print(d1.pvalues.round(4).to_dict())
    print("DIFFS bank+macro R2",round(d2.rsquared,4)); print(d2.pvalues.round(4).to_dict()); print("incremental F",round(Fd,3),round(pFd,4))
    # ---- H3 predictive: bank-only vs bank+macro
    def run(scn,seeds=range(10)):
        rows=[]
        for s in seeds:
            cols=[TARGET]+KEEP
            tr,te=make_split(df,scn,s,cols); mu=tr[cols].mean(); sd=tr[cols].std(ddof=0)
            Ztr=((tr[cols]-mu)/sd); Zte=((te[cols]-mu)/sd); ysd=sd[TARGET]
            for mo in MODELS:
                for lab,fs in [("bank",BANK),("full",BANK+MAC)]:
                    _,p=fit_predict(mo,Ztr[fs].values,Ztr[TARGET].values,Zte[fs].values,s,ysd)
                    rows.append(dict(seed=s,model=mo,spec=lab,RMSE=float(np.sqrt(np.mean((Zte[TARGET].values-p)**2))),R2=float(r2_score(Zte[TARGET].values,p))))
        return pd.DataFrame(rows)
    for scn in ["IQR_random","FULL_random"]:
        r=run(scn); a=r.pivot_table(index=["model"],columns="spec",values=["RMSE","R2"],aggfunc="mean")
        res=[]
        for mo in MODELS:
            b=r[(r.model==mo)&(r.spec=="bank")].sort_values("seed").RMSE.values; f=r[(r.model==mo)&(r.spec=="full")].sort_values("seed").RMSE.values
            try: p=stats.wilcoxon(f,b).pvalue
            except Exception: p=np.nan
            res.append((mo,round(a.loc[mo,("R2","bank")],3),round(a.loc[mo,("R2","full")],3),round(b.mean(),3),round(f.mean(),3),round(100*(f.mean()/b.mean()-1),1),int((f<b).sum()),round(p,3)))
        print(scn); 
        for x in res: print(x)



# ================================================================== COMMAND LINE

def main():
    ap = argparse.ArgumentParser(description="Participatory-finance forecasting study (Morocco, 2018-2026)")
    ap.add_argument("command", choices=["eda", "run", "run_all", "analyze", "hypotheses", "all"])
    ap.add_argument("--data", default="data/Base_Donnees_Finale_Augmentee.xlsx", help="Excel file with sheet 'Donnees_Reelles'")
    ap.add_argument("--out", default="results", help="output folder (tables, figures)")
    ap.add_argument("--work", default="work", help="folder for intermediate .pkl files")
    ap.add_argument("--scenario", default="IQR_random", help="IQR_random | FULL_random | FULL_chrono")
    ap.add_argument("--featset", default="NOBAL", help="NOBAL (8 variables) | ALL17")
    ap.add_argument("--seeds", type=int, default=10, help="number of repetitions")
    a = ap.parse_args()
    configure(a.data, a.out, a.work)
    if a.command in ("eda", "all"): eda()
    if a.command == "run":
        r = run(a.scenario, list(range(a.seeds)), a.featset)
        pickle.dump(r, open(os.path.join(WORK, f"run_{a.scenario}_{a.featset}.pkl"), "wb"))
    if a.command in ("run_all", "all"):
        for sc, fs, ns in [("IQR_random", "NOBAL", a.seeds), ("FULL_random", "NOBAL", a.seeds),
                           ("FULL_chrono", "NOBAL", a.seeds), ("IQR_random", "ALL17", min(5, a.seeds))]:
            r = run(sc, list(range(ns)), fs)
            pickle.dump(r, open(os.path.join(WORK, f"run_{sc}_{fs}.pkl"), "wb"))
    if a.command in ("analyze", "all"): analyze()
    if a.command in ("hypotheses", "all"): hypotheses()


if __name__ == "__main__":
    main()
