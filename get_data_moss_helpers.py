from pathlib import Path
import os
import requests
import geopandas as gpd
from shapely.geometry import box, Point
import folium
import branca.colormap as cm
import numpy as np
from IPython.display import display, HTML
import zipfile
import io
from pathlib import Path


# ============================================================
# Load COMNAP facilities (from latitude/longitude attributes)
# ============================================================


def load_comnap_data():
    url = (
        "https://services7.arcgis.com/tPxy1hrFDhJfZ0Mf/ArcGIS/rest/services/"
        "COMNAP_Antarctic_Facilities_Master/FeatureServer/0/query"
    )

    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "json",
        "returnGeometry": "false",
    }

    resp = requests.get(url, params=params).json()

    rows = []
    for f in resp["features"]:
        a = f.get("attributes", {})
        lat = a.get("Latitude__")
        lon = a.get("Longitude")

        if lat is None or lon is None:
            continue

        rows.append({**a, "geometry": Point(float(lon), float(lat))})

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


# ============================================================
# FULL MAP FUNCTION (moss + COMNAP + ASPA)
# ============================================================

def create_moss_probability_map(
    moss_data,
    species,
    aspa_gdf=None,
    comnap_gdf=None,
    region_bbox=None,
    n_ticks=5
):

    # -----------------------------
    # 0. Species name lookup table
    # -----------------------------
    species_names = {
        "ADep": "Andreaea depressinernis",
        "AGai": "Andreaea gainii",
        "AReg": "Andreaea regularis",
        "BPat": "Bartramia patens",
        "BDry": "Blindia dryptodontoides",
        "BAus": "Brachythecium austrosalebrosum",
        "BAmb": "Bryum amblyodon",
        "BArg": "Bryum argenteum",
        "BPse": "Bryum pseudotriquetrum",
        "CPur": "Ceratodon purpureus",
        "CAci": "Chorisodontium aciphyllum",
        "CLaw": "Coscinodon lawianus",
        "DBra": "Didymodon brachyphyllus",
        "DCap": "Distichium capillaceum",
        "HHei": "Hennediella heimii",
        "PCru": "Pohlia cruda",
        "PNut": "Pohlia nutans",
        "PAlp": "Polytrichastrum alpinum",
        "PPil": "Polytrichum piliferum",
        "PStr": "Polytrichum strictum",
        "SGeo": "Sanionia georgicouncinata",
        "SUnc": "Sanionia uncinata",
        "SGla": "Sarconeurum glaciale",
        "SAnt": "Schistidium antarctici",
        "SFil": "Syntrichia filaris",
        "SPri": "Syntrichia princeps",
        "SSax": "Syntrichia saxicola",
        "WFon": "Warnstorfia fontinaliopsis"
    }

    full_name = species_names.get(species, species)

    # -----------------------------
    # 1. Prepare moss polygons
    # -----------------------------
    moss_data = moss_data.set_crs(epsg=3031, allow_override=True)
    moss = moss_data.to_crs(4326).dropna(subset=[species])
    # Create a formatted display column rounded to 3 d.p.
    moss["display_val"] = moss[species].round(3)

    if region_bbox:
        min_lon, max_lon, min_lat, max_lat = region_bbox
        box_poly = box(min_lon, min_lat, max_lon, max_lat)
        bbox = gpd.GeoDataFrame(geometry=[box_poly], crs=4326)

        try:
            moss = gpd.overlay(moss, bbox, how="intersection")
        except Exception:
            moss = gpd.clip(moss, bbox)

        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2

    else:
        minx, miny, maxx, maxy = moss.total_bounds
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2

    # -----------------------------
    # 2. Prepare ASPA polygons
    # -----------------------------
    aspa_clip = None
    if aspa_gdf is not None:
        aspa_4326 = aspa_gdf.to_crs(4326)

        if region_bbox:
            try:
                aspa_clip = gpd.overlay(aspa_4326, bbox, how="intersection")
            except Exception:
                aspa_clip = gpd.clip(aspa_4326, bbox)
        else:
            aspa_clip = aspa_4326

        aspa_clip = aspa_clip[~aspa_clip.is_empty]

    # -----------------------------
    # 3. Prepare colour scale
    # -----------------------------
    vmin = moss[species].min()
    vmax = moss[species].max()
    colormap = cm.linear.YlGnBu_09.scale(vmin, vmax)

    # -----------------------------
    # 4. Create map
    # -----------------------------
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # --- Google satellite ---
    folium.TileLayer(
        tiles="http://www.google.cn/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satellite"
    ).add_to(m)

    # -----------------------------
    # 5. Moss polygons (tooltip updated)
    # -----------------------------
    folium.GeoJson(
        moss,
        name="Moss probability",
        style_function=lambda f: {
            "fillColor": colormap(f["properties"][species]),
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.7,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["display_val"],
            aliases=[full_name]     # <──── FULL NAME
        )
    ).add_to(m)

    # -----------------------------
    # 6. COMNAP stations
    # -----------------------------
    if comnap_gdf is not None:
        comnap_4326 = comnap_gdf.to_crs(4326)

        if region_bbox:
            comnap_4326 = comnap_4326[
                (comnap_4326.geometry.x >= min_lon) &
                (comnap_4326.geometry.x <= max_lon) &
                (comnap_4326.geometry.y >= min_lat) &
                (comnap_4326.geometry.y <= max_lat)
            ]

        for _, row in comnap_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.geom_type != "Point":
                continue

            folium.CircleMarker(
                location=[geom.y, geom.x],
                radius=5,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.9,
                tooltip=folium.Tooltip(
                    row.get("English_Na", ""),
                    sticky=True
                )
            ).add_to(m)

    # -----------------------------
    # 7. ASPA boundaries (unchanged)
    # -----------------------------
    if aspa_clip is not None and not aspa_clip.empty:
        folium.GeoJson(
            aspa_clip,
            name="ASPA Boundaries",
            style_function=lambda f: {
                "fillColor": "#ff7800",
                "color": "#ff0000",
                "weight": 2,
                "fillOpacity": 0.15,
            },
            tooltip=folium.GeoJsonTooltip(fields=["ASPA_No", "NAME"])
        ).add_to(m)

    # -----------------------------
    # 8. Colour bar (title uses FULL NAME)
    # -----------------------------
    ticks = np.linspace(vmin, vmax, n_ticks)
    labels = [f"{t:.2f}" for t in ticks]
    gradient = ", ".join(colormap(vmin + (vmax - vmin) * i / 100)
                         for i in range(100))

    display(HTML(f"""
        <div style="width:400px;margin-bottom:10px">
            <div style="font-weight:bold;">Probability - {full_name}</div>
            <div style="height:25px;background:linear-gradient(to right,{gradient});border-radius:5px;"></div>
            <div style="display:flex;justify-content:space-between;">
                {''.join(f'<span>{l}</span>' for l in labels)}
            </div>
        </div>
    """))

    folium.LayerControl().add_to(m)
    return m


def create_moss_richness_map(moss_data, aspa_gdf=None, comnap_gdf=None,
                             region_bbox=None, n_ticks=6):
    """
    Creates a folium map showing:
    - Species richness (sum over all species columns)
    - ASPA polygons
    - COMNAP facilities
    """

    # -------------------------------------------
    # Ensure moss data has correct CRS
    # -------------------------------------------
    moss_data = moss_data.set_crs(epsg=3031, allow_override=True)

    # -------------------------------------------
    # Identify species columns (everything except geometry)
    # -------------------------------------------
    species_cols = [
        c for c in moss_data.columns
        if c not in ("geometry", "spatial_ref")
    ]

    if len(species_cols) == 0:
        raise ValueError("No species columns detected in moss dataset.")

    # -------------------------------------------
    # Compute species richness
    # -------------------------------------------
    moss_data["richness"] = moss_data[species_cols].sum(axis=1)

    moss_data["richness_display"] = moss_data["richness"].astype(int)

    # Convert to lat/lon
    moss = moss_data.to_crs(4326)

    # -------------------------------------------
    # Optional bounding box
    # -------------------------------------------
    if region_bbox:
        min_lon, max_lon, min_lat, max_lat = region_bbox
        bbox_poly = box(min_lon, min_lat, max_lon, max_lat)
        bbox = gpd.GeoDataFrame(geometry=[bbox_poly], crs=4326)

        try:
            moss = gpd.overlay(moss, bbox, how="intersection")
        except:
            moss = gpd.clip(moss, bbox)

        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
    else:
        minx, miny, maxx, maxy = moss.total_bounds
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2

    # -------------------------------------------
    # ASPA polygons
    # -------------------------------------------
    aspa_clip = None
    if aspa_gdf is not None:
        aspa_4326 = aspa_gdf.to_crs(4326)

        if region_bbox:
            try:
                aspa_clip = gpd.overlay(aspa_4326, bbox, how="intersection")
            except:
                aspa_clip = gpd.clip(aspa_4326, bbox)
        else:
            aspa_clip = aspa_4326

        aspa_clip = aspa_clip[~aspa_clip.is_empty]

    # -------------------------------------------
    # Colour ramp (integer richness)
    # -------------------------------------------
    vmin = int(moss["richness"].min())
    vmax = int(moss["richness"].max())

    # Round vmax up to nearest 5
    vmax_rounded = int(np.ceil(vmax / 5) * 5)

    colormap = cm.linear.YlGnBu_09.scale(vmin, vmax_rounded)

    # -------------------------------------------
    # Create map
    # -------------------------------------------
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    folium.TileLayer(
        tiles="http://www.google.cn/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satellite"
    ).add_to(m)

    # -------------------------------------------
    # Moss richness polygons
    # -------------------------------------------
    folium.GeoJson(
        moss,
        name="Species Richness",
        style_function=lambda f: {
            "fillColor": colormap(f["properties"]["richness"]),
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=["richness_display"],
                                      aliases=["Species richness"])
    ).add_to(m)

    # -------------------------------------------
    # COMNAP stations
    # -------------------------------------------
    if comnap_gdf is not None:
        comnap_f = comnap_gdf.to_crs(4326)

        if region_bbox:
            min_lon, max_lon, min_lat, max_lat = region_bbox
            comnap_f = comnap_f[
                (comnap_f.geometry.x >= min_lon) &
                (comnap_f.geometry.x <= max_lon) &
                (comnap_f.geometry.y >= min_lat) &
                (comnap_f.geometry.y <= max_lat)
            ]

        for _, row in comnap_f.iterrows():
            g = row.geometry
            if g.geom_type != "Point":
                continue
            folium.CircleMarker(
                location=[g.y, g.x],
                radius=6,
                color=None,        # no outline stroke
                weight=0,
                fill=True,
                fill_color="red",
                fill_opacity=0.9,
                tooltip=row.get("English_Na", "")
            ).add_to(m)

    # -------------------------------------------
    # ASPA boundaries
    # -------------------------------------------
    if aspa_clip is not None:
        folium.GeoJson(
            aspa_clip,
            name="ASPA Boundaries",
            style_function=lambda f: {
                "fillColor": "#ff7800",
                "color": "#ff0000",
                "weight": 2,
                "fillOpacity": 0.15
            },
            tooltip=folium.GeoJsonTooltip(fields=["ASPA_No", "NAME"])
        ).add_to(m)

    # -------------------------------------------
    # Colour bar above map
    # -------------------------------------------
    ticks = np.linspace(vmin, vmax_rounded, n_ticks, dtype=int)
    labels = [str(int(t)) for t in ticks]

    gradient = ", ".join(colormap(vmin + i*(vmax_rounded-vmin)/100)
                         for i in range(100))

    display(HTML(f"""
        <div style="width:450px; margin-bottom:10px; font-family:sans-serif;">
            <b>Moss Species Richness (number of species)</b>
            <div style="height:25px; background:linear-gradient(to right, {gradient}); border-radius:5px;"></div>
            <div style="display:flex; justify-content:space-between;">
                {''.join(f'<span>{l}</span>' for l in labels)}
            </div>
        </div>
    """))

    folium.LayerControl().add_to(m)
    return m


def load_aspa_polygons(local_dir="data_cache"):
    """
    Download ASPA polygons ZIP from GitHub (once) and load it locally.
    Keeps a cached copy on disk to avoid repeated downloads.
    """

    # Where to store the local copy
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    zip_path = local_dir / "ASPAs_polygons_v5_2024.zip"

    # If ZIP is not already downloaded, fetch it
    if not zip_path.exists():
        url = "https://raw.githubusercontent.com/antarcticanz/datamesh-notebooks/main/data/ASPAs_polygons_v5_2024.zip"
        print("Downloading ASPA polygons…")

        resp = requests.get(url)
        resp.raise_for_status()

        with open(zip_path, "wb") as f:
            f.write(resp.content)

    # Read shapefile from the local ZIP
    with zipfile.ZipFile(zip_path, "r") as z:
        shp_name = [name for name in z.namelist() if name.endswith(".shp")][0]
        extract_dir = local_dir / "aspa_polygons_extracted"

        # Extract once
        if not extract_dir.exists():
            z.extractall(extract_dir)

        # Now load using GeoPandas
        shp_path = extract_dir / shp_name
        return gpd.read_file(shp_path)
