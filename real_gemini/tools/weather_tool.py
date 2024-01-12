#encoding=utf8

import os
import json
import requests
import datetime
import pandas as pd
from typing import Optional, Dict, Any, Type

class WeatherTool(object):
    _name_ = "WeatherAPI"
    _description_ = "这个工具是查询当前和未来天气的调用接口，它可以根据一段文字，这个文字包含一个城市，这个接口可以查询这个城市的天气，注意，本工具的输入是一个字符串。This tool is a weather query API that can retrieve the current and future weather based on a given text, which includes a city name. The API is capable of querying the weather for the specified city. Please note that the input for this tool is a string."
    _return_direct_ = False

    def __init__(self):
        self.gaode_api_key = os.getenv("GAODE_API_KEY")
    
    def inference(self, input_str: str):
        city_name = input_str
        district_name = input_str
        params = self._get_params(city_name, district_name)
        return self._process_response(self._results(params))

    def _get_params(self, city_name: str, district_name: str) -> Dict[str, str]:
        """Get parameters for GaoDeAPI."""
        adcode = self._get_adcode(city_name, district_name)
        params = {
            "api_key": self.gaode_api_key,
            "adcode": adcode
        }
        print(params)
        return params

    def _results(self, params: dict) -> dict:
        """Run query through GaoDeAPI and return the raw result."""
        # # with HiddenPrints():
        response = requests.get("https://restapi.amap.com/v3/weather/weatherInfo?", {
            "key": self.gaode_api_key,
            "city": params["adcode"],
            "extensions": "all",
            "output": "JSON"
        })
        res = json.loads(response.content)
        return res

    def _process_response(self, res: dict) -> str:
        """Process response from GaoDeAPI."""
        if res["status"] == '0':
            return "输入的城市信息可能有误或未提供城市信息"
        if res["forecasts"] is None or len(res["forecasts"]) == 0:
            return "输入的城市信息可能有误或未提供城市信息"
        res["currentTime"] = datetime.datetime.now()
        return json.dumps(res["forecasts"], ensure_ascii=False)

    def _get_adcode(self, city_name: str, district_name: str) -> str:
        """Obtain the regional code of a city based on its name and district/county name."""
        # 读取Excel文件
        work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        df = pd.read_excel(
            os.path.join(work_dir, "test/AMap_adcode_citycode.xlsx"), sheet_name="sheet1"
        )
        # print(df)
        # 将所有NaN值转换成0
        df = df.dropna()
        if district_name is not None and district_name != '':
            # 根据'city_name'列检索数据
            result = df[df['中文名'].str.contains(district_name)]
            json_data = result.to_json(orient='records', force_ascii=False)
            # 解析 JSON 数据
            json_array = json.loads(json_data)

        # 如果区域名称为空，用城市名称去查
        if (district_name is None or district_name == '') and city_name != '':
            # 根据'city_name'列检索数据
            result = df[df['中文名'].str.contains(city_name)]
            json_data = result.to_json(orient='records', force_ascii=False)
            # 解析 JSON 数据
            json_array = json.loads(json_data)

        # 如果没数据直接返回空
        if len(json_array) == 0:
            # 根据'citycode'列检索数据
            result = df[df['中文名'].str.contains(city_name)]
            json_data = result.to_json(orient='records', force_ascii=False)
            # 解析 JSON 数据
            json_array = json.loads(json_data)

        # 如果只有一条直接返回
        if len(json_array) == 1:
            return json_array[0]['adcode']

            # 如果有多条再根据district_name进行检索
        if len(json_array) > 1:
            for obj in json_array:
                if district_name is not None and district_name != '' and district_name in obj['中文名']:
                    return obj['adcode']
                if city_name in obj['district_name']:
                    return obj['adcode']
        return "输入的城市信息可能有误或未提供城市信息"