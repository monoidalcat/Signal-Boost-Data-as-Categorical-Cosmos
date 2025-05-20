# Signal-Boost-Data-as-Categorical-Cosmos

**Prologo — Signal Boost: Data as Categorical Cosmos**
Accendere i circuiti di interpretazione: i vostri dati non sono solo numeri, ma **oggetti** in un topos aperto, e ogni trasformazione è una **freccia** che connette stati grezzi a insight rifiniti. Python sarà la nostra **macchina di funtori**, capace di preservare struttura e identità attraverso pipeline ripetute.

---

## 1 Schema Categoriale del Data Workflow

1. **Oggetto Iniziale** (0) → `pd.read_csv(…)`
   Il file CSV è il **vuoto** dal quale emerge un unico morfismo che genera il DataFrame:

   ```python
   df = pd.read_csv("sales.csv")
   ```

   Questo è il solo modo, nella categoria I/O, per ottenere il vettore tabellare.

2. **Identità** (id)

   ```python
   df2 = df.copy()
   ```

   Il **morfismo identità** garantisce che `df2` resti indistinguibile da `df`.

3. **Limite: Prodotto & Pullback**

   * **Prodotto** (`merge`):
     Due DataFrame `df_customers` e `df_sales` fuse sul campo `customer_id` formano

     ```python
     df_full = pd.merge(df_customers, df_sales, on="customer_id", how="inner")
     ```

     che corrisponde al **prodotto fibrato** in cui mantenete **entrambe** le proiezioni sane.
   * **Pullback**:
     Se filtrate `df_sales` per una lista di clienti VIP e poi ricomponete con `df_customers`, ottenete il **pullback**—l’insieme netto di record che soddisfa entrambe le condizioni.

4. **Colimite: Coprodotto & Pushout**

   * **Coprodotto** (`concat`):
     Quando volete appendere più fonti di log,

     ```python
     df_all = pd.concat([df_log1, df_log2], axis=0)
     ```

     create un **coprodotto** che conserva ogni riga distinta.
   * **Pushout**:
     Immaginate due DataFrame con colonne diverse unite da una chiave comune; `merge(..., how="outer")` è il **pushout** che introduce `NaN` dove manca corrispondenza, espandendo il vostro topos di dati.

5. **Mono ed Epi**

   * **Monomorfismo** (`drop_duplicates`):
     Rimuovendo duplicati,

     ```python
     df_unique = df.drop_duplicates(subset="order_id")
     ```

     ottenete una **inclusione netta** di record univoci.
   * **Epimorfismo** (`groupby.agg`):
     Raggruppando e sommando fatturato,

     ```python
     revenue = df.groupby("product_id")["revenue"].sum()
     ```

     collassate più righe in un’unica “qualia” di ricavo per prodotto.

6. **Funtori**

   * **Pandas → NumPy**:
     Convertire le colonne in array NumPy è un **funtore**

     ```python
     arr = df["revenue"].to_numpy()
     ```

     che preserva composizione e identità:

     ```python
     arr2 = df["revenue"].values
     ```

     produce lo stesso array, e `(df["revenue"].to_numpy()) == (df["revenue"].values)` commuta.

7. **Trasformazioni Naturali**

   * Confrontate due pipelines:

     ```python
     clean1 = df.dropna().assign(norm=lambda x: x["revenue"]/x["revenue"].max())
     clean2 = df.assign(norm=lambda x: x["revenue"].fillna(0)/x["revenue"].max()).dropna()
     ```

     Se `clean1.equals(clean2) == True`, avete trovato una **natural transformation** tra i due funtori di pulizia.

8. **Esponenziali**

   * Definite una funzione parametrizzata:

     ```python
     def scale(df, factor):
         return df.assign(scaled=df["value"] * factor)
     ```

     Il **currying** in Python:

     ```python
     from functools import partial
     scale2x = partial(scale, factor=2)
     ```

     Mostra che `scale: (DataFrame, float) → DataFrame` corrisponde a
     `scale~: float → (DataFrame → DataFrame)`, un oggetto esponenziale nel topos di funzioni.

9. **Classificatore Ω**

   * Definite una soglia di outlier:

     ```python
     is_outlier = lambda x: x > df["revenue"].quantile(0.99)
     df["outlier_flag"] = df["revenue"].apply(is_outlier)
     ```

     Qui `outlier_flag: DataFrame → Ω={True,False}` è il **morfismo caratteristico** che taglia il sotto-oggetto degli outlier.

10. **Diagramma Commutativo Finale**
    Visualizzate un confronto tra due modelli di previsione: se due cammini diversi (pre-processing → train → predict) portano allo stesso risultato, il diagramma commuta e garantisce **invarianza** predittiva.

---

**Esempio Completo:**

```python
import pandas as pd
from sklearn.cluster import KMeans

# 1. Carica
df = pd.read_csv("data.csv")

# 2. Pulizia
df = df.dropna(subset=["feature1","feature2"])
df["feature1_norm"] = df["feature1"] / df["feature1"].max()

# 3. Prodotto
df_users = pd.read_csv("users.csv")
df_full = pd.merge(df, df_users, on="user_id", how="inner")

# 4. Coprodotto
df_logs1 = pd.read_csv("log1.csv")
df_logs2 = pd.read_csv("log2.csv")
df_alllogs = pd.concat([df_logs1, df_logs2], axis=0)

# 5. Modello
X = df_full[["feature1_norm","feature2"]]
kmeans = KMeans(n_clusters=3).fit(X)
df_full["cluster"] = kmeans.labels_

# 6. Visualizzazione
import seaborn as sns; import matplotlib.pyplot as plt
sns.scatterplot(data=df_full, x="feature1_norm", y="feature2", hue="cluster")
plt.show()
```

Ogni riga di codice è un **morfismo**, ogni cella un **oggetto**. Le vostre analisi non sono calcoli, ma **diagrammi viventi**: funtori che tessono senso nel caos dei numeri.
