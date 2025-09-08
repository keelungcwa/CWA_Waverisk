import requests
import os
import logging
import time
import json
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from collections import Counter
from datetime import datetime, timedelta

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 從環境變數獲取 API 金鑰
API_KEY = os.getenv("CWA_API_KEY")
if not API_KEY:
    raise ValueError("❌ 環境變數 CWA_API_KEY 未設定！請在 GitHub Secrets 或環境變數中設定")

# 定義儲存目錄（相對路徑）
BASE_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_DIR = os.path.join(os.getcwd(), "docs", "images")
SHAPEFILE_DIR = os.path.join(os.getcwd(), "shapefiles")
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SHAPEFILE_DIR, exist_ok=True)

# 定義檔案路徑和 URL
FORECAST_JSON_FILE = os.path.join(BASE_DIR, "F-D0047-095.json")
RISK_0H_JSON_FILE = os.path.join(BASE_DIR, "F-B0083-005.json")  # 當天00:00
RISK_12H_JSON_FILE = os.path.join(BASE_DIR, "F-B0083-006.json")  # 當天12:00
RISK_24H_JSON_FILE = os.path.join(BASE_DIR, "F-B0083-007.json")  # 次日00:00
SHAPEFILE_PATH = os.path.join(SHAPEFILE_DIR, "TOWN_MOI_1140318.shp")

FORECAST_URL = f"https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/F-D0047-095?Authorization={API_KEY}&downloadType=WEB&format=JSON"
RISK_0H_URL = f"https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/F-B0083-005?Authorization={API_KEY}&downloadType=WEB&format=JSON"
RISK_12H_URL = f"https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/F-B0083-006?Authorization={API_KEY}&downloadType=WEB&format=JSON"
RISK_24H_URL = f"https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/F-B0083-007?Authorization={API_KEY}&downloadType=WEB&format=JSON"

# 北北基桃沿海鄉鎮清單
NORTH_TAIWAN_COASTAL_DISTRICTS = {
    "基隆市": ["中正區", "中山區"],
    "新北市": ["貢寮區", "瑞芳區", "萬里區", "金山區", "石門區", "三芝區", "淡水區", "八里區", "林口區"],
    "桃園市": ["蘆竹區", "大園區", "觀音區", "新屋區"],
    "台北市": []
}

# 測站到區的映射
STATION_TO_DISTRICT = {
    "基隆碧砂": ("基隆市", "中正區"),
    "新北龍洞": ("新北市", "貢寮區"),
    "新北野柳": ("新北市", "萬里區"),
    "桃園永安": ("桃園市", "新屋區")
}

# 測站與港口經緯度
STATION_COORDINATES = {
    "基隆碧砂": (121.78639, 25.14667),
    "新北龍洞": (121.9220, 25.1092),
    "新北野柳": (121.68853, 25.20791),
    "桃園永安": (121.02258, 24.98773),
    "基隆港": (121.7430, 25.1310),
    "臺北港": (121.3550, 25.1500)
}

# 風險值與顏色映射
RISK_MAPPING = {
    0: "低（浪高0-1.5公尺或風險低）",
    1: "中（浪高1.5-3公尺或風險中）",
    2: "高（浪高3-6公尺或風險高）",
    3: "極高（浪高>=6公尺）"
}
COLOR_MAPPING = {
    0: "#00A65180",
    1: "#FFF20080",
    2: "#ED1C2480",
    3: "#EE00EE80"
}

def download_file(url, file_path, retries=1, timeout=10):
    for attempt in range(retries + 1):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"成功下載並覆蓋檔案到: {file_path}")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"下載檔案失敗 {url} (嘗試 {attempt + 1}/{retries + 1}): {e}")
            if attempt < retries:
                logging.info(f"等待 5 秒後重試...")
                time.sleep(5)
        except IOError as e:
            logging.error(f"寫入檔案失敗 {file_path}: {e}")
            break
    return False

def parse_numeric(value):
    try:
        return float(value) if value != "-" else None
    except:
        return None

def parse_integer(value):
    try:
        return int(value) if value != "-" else None
    except:
        return None

def parse_direction(values):
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return "無資料"
    counter = Counter(valid_values)
    return max(counter, key=lambda x: (counter[x], x))

def load_wave_risk_data(file_paths):
    risk_data = []
    for file_path, time_label in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            locations = data["cwaopendata"]["dataset"]["Location"]
            for loc in locations:
                station_name = loc["LocationName"]
                if station_name in STATION_TO_DISTRICT:
                    lon, lat = STATION_COORDINATES.get(station_name, (None, None))
                    if lon is None or lat is None:
                        logging.warning(f"{station_name} 無經緯度資料，跳過")
                        continue
                    risk_data.append({
                        "station_name": station_name,
                        "county": STATION_TO_DISTRICT[station_name][0],
                        "district": STATION_TO_DISTRICT[station_name][1],
                        f"risk_value_{time_label}": int(loc["RiskValue"]),
                        "longitude": lon,
                        "latitude": lat
                    })
        except Exception as e:
            logging.error(f"載入 {file_path} ({time_label}) 失敗: {e}")
    df = pd.DataFrame(risk_data)
    if df.empty:
        return None
    df = df.groupby(['station_name', 'county', 'district', 'longitude', 'latitude']).agg({
        'risk_value_0h': 'first',
        'risk_value_12h': 'first',
        'risk_value_24h': 'first'
    }).reset_index()
    for col in ['risk_value_0h', 'risk_value_12h', 'risk_value_24h']:
        df[col] = df[col].fillna(99).astype(int)
    df["max_risk"] = df[['risk_value_0h', 'risk_value_12h', 'risk_value_24h']].replace(99, -1).max(axis=1)
    df["max_risk_label"] = df["max_risk"].apply(lambda x: "尚未建置" if x == -1 else RISK_MAPPING[x].split("（")[0])
    return df

def load_forecast_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = today.strftime("%Y-%m-%dT%H:%M:%S+08:00")
        end_time = (today + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S+08:00")
        
        forecast_data = []
        locations = data["cwaopendata"]["Dataset"]["Locations"]["Location"]
        
        target_districts = []
        for districts in NORTH_TAIWAN_COASTAL_DISTRICTS.values():
            target_districts.extend(districts)
        
        for loc in locations:
            location_name = loc["LocationName"]
            if ("沿海" in location_name and 
                any(district in location_name for district in target_districts) and 
                "彭佳嶼" not in location_name):
                for city, districts in NORTH_TAIWAN_COASTAL_DISTRICTS.items():
                    for district in districts:
                        if district in location_name:
                            county = city
                            break
                    else:
                        continue
                    break
                else:
                    continue
                
                lon = float(loc["Longitude"])
                lat = float(loc["Latitude"])
                
                data_entry = {
                    "location_name": location_name,
                    "county": county,
                    "district": district,
                    "longitude": lon,
                    "latitude": lat,
                    "wind_speed": [],
                    "beaufort_scale": [],
                    "wind_direction": [],
                    "wave_height": [],
                    "wave_direction": [],
                    "wave_period": [],
                    "ocean_current_speed": [],
                    "ocean_current_direction": []
                }
                
                for element in loc["WeatherElement"]:
                    element_name = element["ElementName"]
                    for time_data in element["Time"]:
                        data_time = time_data["DataTime"]
                        if start_time <= data_time <= end_time:
                            value = time_data["ElementValue"]
                            if element_name == "風速":
                                data_entry["wind_speed"].append(parse_numeric(value["WindSpeed"]))
                                data_entry["beaufort_scale"].append(parse_integer(value.get("BeaufortScale")))
                            elif element_name == "風向":
                                data_entry["wind_direction"].append(parse_direction([value["WindDirection"]]))
                            elif element_name == "浪高":
                                data_entry["wave_height"].append(parse_numeric(value["WaveHeight"]))
                            elif element_name == "浪向":
                                data_entry["wave_direction"].append(parse_direction([value["WaveDirection"]]))
                            elif element_name == "浪週期":
                                data_entry["wave_period"].append(parse_numeric(value["WavePeriod"]))
                            elif element_name == "流速":
                                data_entry["ocean_current_speed"].append(parse_numeric(value["OceanCurrentSpeed"]))
                            elif element_name == "流向":
                                data_entry["ocean_current_direction"].append(parse_direction([value["OceanCurrentDirection"]]))
                
                max_wave_height = max([x for x in data_entry['wave_height'] if x is not None], default=0)
                alert_level = (
                    3 if max_wave_height >= 6
                    else 2 if max_wave_height > 3
                    else 1 if max_wave_height >= 1.5
                    else 0
                )
                
                wind_speed_range = "無資料"
                if data_entry['wind_speed']:
                    rounded_values = [round(x) for x in data_entry['wind_speed'] if x is not None]
                    min_val = min(rounded_values)
                    max_val = max(rounded_values)
                    wind_speed_range = f"{min_val}" if min_val == max_val else f"{min_val}~{max_val}"
                
                beaufort_scale_range = "無資料"
                if data_entry['beaufort_scale']:
                    values = [x for x in data_entry['beaufort_scale'] if x is not None]
                    min_val = min(values)
                    max_val = max(values)
                    beaufort_scale_range = f"{min_val}" if min_val == max_val else f"{min_val}~{max_val}"
                
                wave_height_range = "無資料"
                if data_entry['wave_height']:
                    rounded_values = [round(x, 1) for x in data_entry['wave_height'] if x is not None]
                    min_val = min(rounded_values)
                    max_val = max(rounded_values)
                    wave_height_range = f"{min_val:.1f}" if min_val == max_val else f"{min_val:.1f}~{max_val:.1f}"
                
                wave_period_range = "無資料"
                if data_entry['wave_period']:
                    rounded_values = [round(x) for x in data_entry['wave_period'] if x is not None]
                    min_val = min(rounded_values)
                    max_val = max(rounded_values)
                    wave_period_range = f"{min_val}" if min_val == max_val else f"{min_val}~{max_val}"
                
                ocean_current_speed_range = "無資料"
                if data_entry['ocean_current_speed']:
                    rounded_values = [round(x, 1) for x in data_entry['ocean_current_speed'] if x is not None]
                    min_val = min(rounded_values)
                    max_val = max(rounded_values)
                    ocean_current_speed_range = f"{min_val:.1f}" if min_val == max_val else f"{min_val:.1f}~{max_val:.1f}"
                
                forecast_data.append({
                    "location_name": location_name,
                    "county": county,
                    "district": district,
                    "longitude": lon,
                    "latitude": lat,
                    "wind_speed_range": wind_speed_range,
                    "beaufort_scale_range": beaufort_scale_range,
                    "wind_direction": parse_direction(data_entry["wind_direction"]),
                    "wave_height_range": wave_height_range,
                    "wave_direction": parse_direction(data_entry["wave_direction"]),
                    "wave_period_range": wave_period_range,
                    "ocean_current_speed_range": ocean_current_speed_range,
                    "ocean_current_direction": parse_direction(data_entry["ocean_current_direction"]),
                    "alert_level": alert_level
                })
        
        df = pd.DataFrame(forecast_data)
        all_districts = [
            ("基隆市", "中正區"), ("基隆市", "中山區"),
            ("新北市", "貢寮區"), ("新北市", "瑞芳區"), ("新北市", "萬里區"),
            ("新北市", "金山區"), ("新北市", "石門區"), ("新北市", "三芝區"),
            ("新北市", "淡水區"), ("新北市", "八里區"), ("新北市", "林口區"),
            ("桃園市", "蘆竹區"), ("桃園市", "大園區"), ("桃園市", "觀音區"),
            ("桃園市", "新屋區")
        ]
        for county, district in all_districts:
            if district not in df["district"].values:
                df = pd.concat([df, pd.DataFrame([{
                    "location_name": f"{district}沿海",
                    "county": county,
                    "district": district,
                    "longitude": None,
                    "latitude": None,
                    "wind_speed_range": "無資料",
                    "beaufort_scale_range": "無資料",
                    "wind_direction": "無資料",
                    "wave_height_range": "無資料",
                    "wave_direction": "無資料",
                    "wave_period_range": "無資料",
                    "ocean_current_speed_range": "無資料",
                    "ocean_current_direction": "無資料",
                    "alert_level": 0
                }])], ignore_index=True)
        
        county_order = {"基隆市": 0, "新北市": 1, "桃園市": 2}
        district_order = {
            "中正區": 0, "中山區": 1,
            "貢寮區": 0, "瑞芳區": 1, "萬里區": 2, "金山區": 3, "石門區": 4,
            "三芝區": 5, "淡水區": 6, "八里區": 7, "林口區": 8,
            "蘆竹區": 0, "大園區": 1, "觀音區": 2, "新屋區": 3
        }
        df["county_order"] = df["county"].map(county_order)
        df["district_order"] = df["district"].map(district_order)
        df = df.sort_values(["county_order", "district_order"]).drop(columns=["county_order", "district_order"])
        
        sent_time = data["cwaopendata"]["Sent"]
        sent_date = sent_time.split("T")[0]
        start_time = f"{sent_date} 00:00"
        end_date = (datetime.strptime(sent_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y/%m/%d")
        forecast_time = f"{start_time}~{end_date} 00:00"
        return df, forecast_time
    except Exception as e:
        logging.error(f"載入預報 JSON 失敗: {e}")
        return None, None

def aggregate_risk_by_district(forecast_df, risk_df):
    district_risk = {}
    for district in NORTH_TAIWAN_COASTAL_DISTRICTS["基隆市"] + NORTH_TAIWAN_COASTAL_DISTRICTS["新北市"] + NORTH_TAIWAN_COASTAL_DISTRICTS["桃園市"]:
        district_forecast = forecast_df[forecast_df["district"] == district]
        alert_level = district_forecast["alert_level"].max() if not district_forecast.empty else 0
        district_risk_data = risk_df[risk_df["district"] == district]
        max_risk = district_risk_data["max_risk"].max() if not district_risk_data.empty else -1
        final_risk = max(alert_level, max_risk) if max_risk != -1 else alert_level
        district_risk[district] = final_risk
    return pd.DataFrame(list(district_risk.items()), columns=["district", "risk_value"])

def load_taiwan_map():
    try:
        gdf = gpd.read_file(SHAPEFILE_PATH, encoding='utf-8')
        if gdf.geometry.is_empty.any() or gdf.geometry.isna().any():
            raise ValueError("無效幾何資料")
        if 'TOWNNAME' not in gdf.columns:
            raise ValueError(f"shapefile 缺少 'TOWNNAME' 欄位，可用欄位：{gdf.columns.tolist()}")
        north_taiwan_counties = ["基隆市", "新北市", "台北市", "桃園市"]
        gdf = gdf[gdf["COUNTYNAME"].isin(north_taiwan_counties)]
        gdf = gdf.dissolve(by='TOWNNAME', as_index=False)
        if gdf.geometry.is_empty.any() or gdf.geometry.isna().any():
            raise ValueError("聚合後無效幾何資料")
        coastal_districts = []
        for districts in NORTH_TAIWAN_COASTAL_DISTRICTS.values():
            coastal_districts.extend(districts)
        gdf["is_coastal"] = gdf["TOWNNAME"].apply(lambda x: x in coastal_districts)
        return gdf
    except Exception as e:
        logging.error(f"載入 shapefile 失敗: {e}")
        raise

def plot_north_taiwan_map(forecast_df, risk_df, taiwan_gdf, district_risk_df, forecast_time):
    north_taiwan_stations = ["基隆碧砂", "新北龍洞", "新北野柳", "桃園永安"]
    ports = ["基隆港", "臺北港"]
    
    taiwan_gdf = taiwan_gdf.merge(district_risk_df[['district', 'risk_value']].drop_duplicates(), how="left", left_on="TOWNNAME", right_on="district")
    taiwan_gdf["risk_value"] = taiwan_gdf["risk_value"].fillna(-1).astype(int)
    
    colors = [COLOR_MAPPING[i] for i in [0, 1, 2, 3]]
    cmap = ListedColormap(colors)
    
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax_map, ax_table) = plt.subplots(1, 2, figsize=(14, 7), dpi=100, gridspec_kw={'width_ratios': [1, 1.2]})
    fig.subplots_adjust(wspace=0.05)
    
    taiwan_gdf[taiwan_gdf["is_coastal"]].plot(
        ax=ax_map,
        column="risk_value",
        cmap=cmap,
        edgecolor='black',
        linewidth=0.5,
        legend=False,
        missing_kwds={'color': 'white', 'edgecolor': 'black', 'linewidth': 0.5},
        vmin=0,
        vmax=3
    )
    taiwan_gdf[~taiwan_gdf["is_coastal"]].plot(
        ax=ax_map,
        facecolor='white',
        edgecolor='black',
        linewidth=0.5
    )
    
    for idx, row in risk_df.iterrows():
        if row["station_name"] in north_taiwan_stations:
            if pd.notna(row["longitude"]) and pd.notna(row["latitude"]):
                district = row["district"]
                final_risk = district_risk_df[district_risk_df["district"] == district]["risk_value"].iloc[0]
                if final_risk != -1:
                    ax_map.scatter(
                        row["longitude"],
                        row["latitude"],
                        s=150,
                        facecolor=COLOR_MAPPING[final_risk],
                        edgecolor='black',
                        linewidth=1.5
                    )
                    ax_map.annotate(
                        row["station_name"],
                        xy=(row["longitude"], row["latitude"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        fontfamily='Noto Sans CJK TC',
                        color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                    )
    
    for port in ports:
        lon, lat = STATION_COORDINATES.get(port, (None, None))
        if lon is not None and lat is not None:
            ax_map.scatter(
                lon,
                lat,
                s=15,
                facecolor='black',
                edgecolor='black',
                linewidth=0.5
            )
            ax_map.annotate(
                port,
                xy=(lon, lat),
                xytext=(-10, -5),
                textcoords="offset points",
                fontsize=8,
                fontfamily='Noto Sans CJK TC',
                color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
    
    ax_map.set_xlim(121.0, 122.0)
    ax_map.set_ylim(24.5, 25.5)
    ax_map.set_axis_off()
    ax_map.set_aspect('auto')
    
    legend_elements = [
        Circle((0, 0), radius=0.04, facecolor=COLOR_MAPPING[i], edgecolor='black', linewidth=1.5, label=RISK_MAPPING[i])
        for i in [3, 2, 1, 0]
    ]
    ax_map.legend(handles=legend_elements, loc="upper left", title="風險等級", prop={'family': 'Noto Sans CJK TC', 'size': 9.6}, handlelength=1.2, handleheight=1.2)
    
    fig.suptitle("基隆北海岸(北北基桃)沿海與異常波浪(瘋狗浪)預報", fontsize=14, fontfamily='Noto Sans CJK TC', x=0.5, y=0.98)
    fig.text(0.5, 0.92, f"預報時間: {forecast_time}", fontsize=10, fontfamily='Noto Sans CJK TC', ha='center')
    
    forecast_df = forecast_df.merge(risk_df[["district", "max_risk", "max_risk_label"]], on="district", how="left")
    forecast_df["max_risk"] = forecast_df["max_risk"].fillna(-1).astype(int)
    forecast_df["max_risk_label"] = forecast_df["max_risk_label"].fillna("尚未建置")
    forecast_df["final_risk"] = forecast_df.apply(lambda row: max(row["alert_level"], row["max_risk"]) if row["max_risk"] != -1 else row["alert_level"], axis=1)
    
    table_data = [
        [
            row["county"],
            row["district"],
            row["wind_speed_range"],
            row["beaufort_scale_range"],
            row["wind_direction"],
            row["wave_height_range"],
            row["wave_direction"],
            row["wave_period_range"],
            row["max_risk_label"],
            row["ocean_current_speed_range"],
            row["ocean_current_direction"]
        ]
        for _, row in forecast_df.iterrows()
    ]
    
    table = ax_table.table(
        cellText=[
            ["縣市", "區", "風速\n(公尺/秒)", "風級", "風向", "浪高\n(公尺)", "浪向", "浪週期\n(秒)", "異常波浪\n(瘋狗浪)風險", "流速\n(公尺/秒)", "流向"]
        ] + table_data,
        cellLoc='center',
        loc='upper center',
        colWidths=[0.12, 0.12, 0.12, 0.1, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_height(0.12)
    
    for table_row_idx, row in enumerate(forecast_df.iterrows()):
        _, row_data = row
        if row_data["final_risk"] in [1, 2, 3]:
            for col in range(11):
                table[(table_row_idx + 1, col)].set_facecolor(COLOR_MAPPING[row_data["final_risk"]])
    
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(fontfamily='Noto Sans CJK TC')
        if row == 0:
            cell.set_text_props(weight='bold')
    ax_table.axis('off')
    
    ax_table.text(
        0.95, 0.09,
        f"中央氣象署 基隆氣象站 製圖\n{forecast_time.split(' ')[0]}\n資料來源：氣象署 鄉鎮沿海預報+異常浪(瘋狗浪)風險預報",
        fontsize=10, fontfamily='Noto Sans CJK TC',
        verticalalignment='bottom', horizontalalignment='right'
    )
    
    # 計算前一天日期
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    output_filename = os.path.join(OUTPUT_DIR, f"CWA_waverisk_{previous_date}.png")
    fixed_filename = os.path.join(OUTPUT_DIR, "CWA_waverisk.png")
    
    # 如果固定檔案名已存在，將其改名為前一天日期
    if os.path.exists(fixed_filename):
        try:
            os.rename(fixed_filename, output_filename)
            logging.info(f"已將 {fixed_filename} 改名為 {output_filename}")
        except Exception as e:
            logging.error(f"改名 {fixed_filename} 失敗: {e}")
    
    # 儲存新圖表
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(fixed_filename, dpi=100, bbox_inches='tight')
    logging.info(f"圖表已儲存至: {fixed_filename}")
    plt.close()

def main():
    current_time = datetime.now().strftime("%Y/%m/%d")
    
    # 下載檔案
    files_to_download = [
        (FORECAST_URL, FORECAST_JSON_FILE, "鄉鎮天氣預報檔案"),
        (RISK_0H_URL, RISK_0H_JSON_FILE, "異常波浪風險檔案（當天00:00）"),
        (RISK_12H_URL, RISK_12H_JSON_FILE, "異常波浪風險檔案（當天12:00）"),
        (RISK_24H_URL, RISK_24H_JSON_FILE, "異常波浪風險檔案（次日00:00）")
    ]
    
    for url, file_path, description in files_to_download:
        if not download_file(url, file_path):
            logging.warning(f"無法下載 {description}，請檢查網路連線或URL")
    
    # 檢查 shapefile 是否存在
    if not os.path.exists(SHAPEFILE_PATH):
        logging.error(f"shapefile 不存在: {SHAPEFILE_PATH}")
        return
    
    # 檢查 shapefile 相關檔案
    required_extensions = ['.shp', '.shx', '.dbf', '.prj']
    for ext in required_extensions:
        if not os.path.exists(os.path.join(SHAPEFILE_DIR, f"TOWN_MOI_1140318{ext}")):
            logging.error(f"缺少 shapefile 相關檔案: {SHAPEFILE_DIR}/TOWN_MOI_1140318{ext}")
            return
    
    # 處理資料並生成圖表
    file_paths = [
        (RISK_0H_JSON_FILE, '0h'),
        (RISK_12H_JSON_FILE, '12h'),
        (RISK_24H_JSON_FILE, '24h')
    ]
    risk_df = load_wave_risk_data(file_paths)
    if risk_df is None or risk_df.empty:
        logging.warning("無可用風險資料，請確保 JSON 檔案存在且格式正確")
        risk_df = pd.DataFrame(columns=["district", "max_risk", "max_risk_label"])
    
    forecast_df, forecast_time = load_forecast_data(FORECAST_JSON_FILE)
    if forecast_df is None or forecast_df.empty:
        logging.error("無可用預報資料，請確保 JSON 檔案存在且格式正確")
        return
    if forecast_time is None:
        forecast_time = f"{current_time}~{(datetime.now() + timedelta(days=1)).strftime('%Y/%m/%d')} 00:00"
    
    district_risk_df = aggregate_risk_by_district(forecast_df, risk_df)
    taiwan_gdf = load_taiwan_map()
    plot_north_taiwan_map(forecast_df, risk_df, taiwan_gdf, district_risk_df, forecast_time)

if __name__ == "__main__":
    main()
