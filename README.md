# DBSCAN & HDBSCAN clustering explorer

A Streamlit app that turns the notebook **DBScan_HDBScan.ipynb** ("Comparing DBSCAN and HDBSCAN clustering",
ODCAF museums across Canada) into a general tool: the same pipeline, for any CSV file.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Put `app.py` next to the notebook's `data/` folder to get the notebook's dataset and basemap preloaded:

```
app.py
requirements.txt
data/
  ODCAF_V1-0/ODCAF_v1-0.csv   <- opened by default; the defaults reproduce the notebook
  Canada.tif                  <- basemap of the notebook-style static map
  any_other_file.csv          <- every CSV / TSV / .csv.gz / .zip in data/ is listed in the sidebar
```

Without a `data/` folder, upload a file or paste a link in the sidebar.

## From the notebook to the app

| Notebook | App |
|---|---|
| `pd.read_csv(url, encoding="ISO-8859-1")` | **Sidebar**: upload, URL or `data/` folder. Encoding, delimiter, decimal separator and header row are detected; `.zip` and `.gz` archives are opened |
| `df.head()`, `df.info()`, `value_counts()`, `'..'` as missing values | **Data** tab; missing-value markers are configurable (default `..` and `?`) |
| `df[df.ODCAF_Facility_Type == 'museum']` | **Prepare › Filter**: value lists or numeric ranges on any columns |
| `df[['Latitude', 'Longitude']]`, `astype('float')` | **Prepare › Features**: any numeric columns (text numbers such as `49.89,` are parsed), optional one-hot categorical columns, incomplete rows dropped |
| `coords_scaled['Latitude'] = 2 * ...` | **Prepare › Distance and scaling**: none / z-score / min-max / robust, per-feature weights (latitude ×2 by default for coordinates), Euclidean, Manhattan, Chebyshev or haversine (km) |
| `DBSCAN(eps=1.0, min_samples=3)` | **DBSCAN** tab, with a k-distance plot and a suggested eps (it suggests 1.0 on the museums) |
| `hdbscan.HDBSCAN(min_cluster_size=3, min_samples=None)` | **HDBSCAN** tab: hdbscan library or scikit-learn, plus `cluster_selection_epsilon`, `leaf`/`eom`, `allow_single_cluster` |
| `df['Cluster'].value_counts()` | Cluster size table and chart, cluster profiles, quality metrics |
| `plot_clustered_locations(...)` on `Canada.tif` | Interactive map for any region (or PCA / feature scatter plots for non-geographic data) and the same static matplotlib map on a GeoTIFF or online tiles |
| Closing questions | **Compare** tab: side-by-side metrics, agreement scores, who is noise, how HDBSCAN splits DBSCAN's large clusters |
| | **Export** tab: the clustered CSV and a standalone Python script that reproduces the run |

## Reproducing the notebook

With the ODCAF file loaded, the defaults are the notebook's choices: museums only, Latitude/Longitude,
latitude × 2, Euclidean distance, DBSCAN `eps=1.0, min_samples=3`, HDBSCAN `min_cluster_size=3, min_samples=None`.

* DBSCAN gives exactly the notebook's result: 1,607 museums, 33 clusters, 79 noise points, largest cluster 701.
* HDBSCAN gives about 140 clusters and about 465 noise points. The exact numbers move by a few between hdbscan
  versions (the notebook shows 140 clusters / 464 noise) because the museums contain duplicate coordinates, and
  ties are broken differently from one release to the next.
* scikit-learn counts the point itself in `min_samples`; the hdbscan library does not. With the scikit-learn
  implementation the app passes `min_samples + 1`, so both follow the notebook's convention.

## Notes

* Interactive map tiles come from the internet. Offline, pick **No basemap** in the sidebar.
* Above 20,000 rows the app offers to cluster a random sample (on by default above 50,000), and it refuses an eps
  so large that DBSCAN would run out of memory.
* The exported script repeats the app's reading options, filters, sampling, scaling and parameters, so it
  reproduces the app's labels.
