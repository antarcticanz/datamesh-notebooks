from IPython.display import display, HTML
import numpy as np
import branca.colormap as cm
from shapely.geometry import box
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import requests
import zipfile
import os


def download_and_extract_lima(url="https://lima.usgs.gov/tiff_90pct.zip",
                              zip_name="tiff_90pct.zip",
                              extract_dir="tiff_90pct",
                              specific_tif="tiff_90pct/00000-20080319-092059124.tif"):
    """
    Download a zipped LIMA dataset, extract it, and return path to a specific GeoTIFF.

    Parameters:
    - url: URL of the zip file
    - zip_name: name to save the downloaded zip locally
    - extract_dir: directory to extract the files
    - specific_tif: relative path inside the zip to a specific GeoTIFF

    Returns:
    - tif_path: full path to the requested GeoTIFF
    """
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

    # 1. Download
    zip_path = os.path.join(cwd, zip_name)
    print(f"Downloading file to: {zip_path} ...")
    response = requests.get(url)
    with open(zip_path, "wb") as f:
        f.write(response.content)
    print("Download complete.")

    # 2. Extract
    extract_path = os.path.join(cwd, extract_dir)
    print(f"Extracting files to: {extract_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")

    # 3. Build path to specific GeoTIFF
    tif_path = os.path.join(extract_path, specific_tif)
    print(f"GeoTIFF path: {tif_path}")

    print("LIMA base map downloaded successfully.")
    return tif_path


def plot_species_on_lima(region_bbox, species_col,
                         lima_path=r"C:\Users\ANTNZDEV\michaelmeredythyoung\mapping3\lima\00000-20080319-092059124.tif",
                         shapefile_path=r"C:\Users\ANTNZDEV\OneDrive - Antarctica New Zealand\Documents\ANTNZDEV\github_dashboard_products\Moss-Shiny-App\all_pred_prob.shp"):
    """
    Plot LIMA raster with Antarctic species probabilities from shapefile.

    Parameters
    ----------
    region_bbox : tuple
        (xmin, xmax, ymin, ymax) in meters for the area to zoom in.
    species_col : str
        Column name in shapefile representing the species probability.
    lima_path : str
        Path to LIMA raster file.
    shapefile_path : str
        Path to Antarctic shapefile with species probability columns.
    """

    # -----------------------
    # 1. Load LIMA raster
    # -----------------------
    with rasterio.open(lima_path) as src:
        lima_data = src.read([1, 2, 3])  # RGB bands
        lima_extent = (src.bounds.left, src.bounds.right,
                       src.bounds.bottom, src.bounds.top)
        lima_crs = src.crs

    lima_img = lima_data.transpose(1, 2, 0)  # rows, cols, RGB

    # -----------------------
    # 2. Load shapefile
    # -----------------------
    gdf = gpd.read_file(shapefile_path)

    # -----------------------
    # 3. Reproject shapefile to match raster CRS
    # -----------------------
    gdf = gdf.to_crs(lima_crs)

    # -----------------------
    # 4. Plot raster and shapefile
    # -----------------------
    xmin, xmax, ymin, ymax = region_bbox
    plt.figure(figsize=(12, 12))
    plt.imshow(lima_img, extent=lima_extent)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # Plot shapefile polygons colored by species probabilities
    gdf.plot(
        ax=plt.gca(),
        column=species_col,
        cmap="viridis",
        alpha=0.6,
        edgecolor="black",
        linewidth=0.5,
        legend=True
    )

    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title(f"LIMA Raster with {species_col} Probability Overlay")
    plt.show()


def create_probability_map(shapefile_path, species, region_bbox=None, n_ticks=5):
    """
    Create an interactive Folium probability map with:
    - Continuous gradient colour bar outside the map
    - Numeric tick values in black
    - Label on top of the colour bar
    - Google satellite basemap
    """
    # 1. Load shapefile
    gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    if species not in gdf.columns:
        raise ValueError(f"Column '{species}' not found in shapefile")
    gdf = gdf[~gdf[species].isna()].copy()
    if gdf.empty:
        raise ValueError("No valid polygons to plot")

    # 2. Clip to bounding box if provided
    if region_bbox is not None:
        xmin, xmax, ymin, ymax = region_bbox
        bbox_gdf = gpd.GeoDataFrame(
            geometry=[box(xmin, ymin, xmax, ymax)], crs=4326)
        try:
            gdf = gpd.overlay(gdf, bbox_gdf, how="intersection")
        except Exception:
            gdf = gpd.clip(gdf, bbox_gdf)
        if gdf.empty:
            raise ValueError("No polygons intersect the bounding box")
        center_lon = (xmin + xmax)/2
        center_lat = (ymin + ymax)/2
    else:
        minx, miny, maxx, maxy = gdf.total_bounds
        center_lon = (minx + maxx)/2
        center_lat = (miny + maxy)/2

    # 3. Colour scale
    vmin = float(np.nanmin(gdf[species]))
    vmax = float(np.nanmax(gdf[species]))
    colormap = cm.linear.YlGnBu_09.scale(vmin, vmax)

    # 4. Create Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
    folium.TileLayer(
        tiles="http://www.google.cn/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite"
    ).add_to(m)

    # 5. Colour polygons
    def style_function(feature):
        val = feature["properties"].get(species, None)
        try:
            color = colormap(float(val))
        except:
            color = "#00000000"
        return {"fillColor": color, "color": "black", "weight": 0.5, "fillOpacity": 0.7}

    folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=[species], aliases=[
                                      f"{species}:"], sticky=True)
    ).add_to(m)
    folium.LayerControl().add_to(m)

    # 6. External continuous gradient with numeric black labels
    tick_values = np.linspace(vmin, vmax, n_ticks)
    tick_labels = [f"{t:.2f}" for t in tick_values]

    # High-resolution gradient for smooth display
    gradient_positions = np.linspace(0, 100, 256)
    gradient_colors = [colormap(vmin + (vmax-vmin)*p/100)
                       for p in gradient_positions]
    gradient_css = "linear-gradient(to right, " + \
        ", ".join(gradient_colors) + ")"

    # Build HTML for colourbar with label on top
    html = f"""
    <div style="width:400px; margin-bottom:10px; font-family:sans-serif;">
        <div style="color:black; font-weight:bold; margin-bottom:5px;">{species} probability</div>
        <div style="height:25px; border-radius:5px; background:{gradient_css};"></div>
        <div style="display:flex; justify-content:space-between; color:black; font-weight:bold; margin-top:3px;">
            {''.join(f'<span>{label}</span>' for label in tick_labels)}
        </div>
    </div>
    """

    # Display in Jupyter notebook
    display(HTML(html))
    display(m)
    return m, colormap
