# -*- coding: utf-8 -*-
import argparse
import requests
import json
import os
import re
import zipfile

try:
	import pandas as pd
except ImportError:
	pd = None

class Url:
	def __init__(self):
		self.host_list = ['.58corp.com','aihouse.58corp.com','passport.58corp.com']
		self.user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
		self.login_url = 'https://passport.58corp.com/login?service=https%3A%2F%2Faihouse.58corp.com%2F'
		self.aihouse_url = "https://aihouse.58corp.com/report/byid"
		# 完整 Cookie 头字符串；None 时首次请求会调用 readCookies() 填充。也可在运行期直接赋值以动态更新。
		self.cookie_string = 'xxzlclientid=727a7e1d-08d2-47df-b164-1774582600521; xxzlxxid=pfmxbuGWwvZwEB4IxXDh0xpmec1F5H9At+0Hc6PEXO9nM75OzISzxl1oJp61soSV3S3g; xxzlbbid=pfmbRLroKy6LV9IcmyzqUYHsCt6N1loWpFzHkBUeu8+CL0nxGIKVGgkuerpN2dCgNmSZJ7uU/Auv6ncLzJ1B7/ZEgLCJtCvK5P9G6zT9I9cmShSuVj015MVUJ0RcswLKRDbm55FV/yIxNzc0NTgyNjAyOTMxODkz_1; dunCookie=69db01dcd6a19623cf0df9dedb0203ebc98397e67365e29bcffb6f75b2301d5b4e81cc4624e41c73; SSO_SESSION_ID=ST-1645978-IX9IbhSeSUchGNEdNgas-passport-58corp-com'
		self.headers = None

	@staticmethod
	def parse_cookie_string(cookie_header):
		"""
		将 "a=1; b=2" 解析为字典（值中可含 = 号，按第一个 = 分割键值）。
		"""
		out = {}
		if not cookie_header or not str(cookie_header).strip():
			return out
		for part in str(cookie_header).split(';'):
			part = part.strip()
			if not part:
				continue
			if '=' not in part:
				continue
			k, v = part.split('=', 1)
			out[k.strip()] = v.strip()
		return out

	def getCookiesStr(self, name_list):
		"""
		从当前 cookie 字符串中按 name_list 顺序取出若干项，拼成请求用的 Cookie 头。
		缺少的键会跳过（不报错）。
		"""
		if self.cookie_string is None:
			self.cookie_string = self.readCookies()
		parsed = self.parse_cookie_string(self.cookie_string)
		parts = []
		for key in name_list:
			if key in parsed:
				parts.append(key + '=' + parsed[key])
		return '; '.join(parts)

	def readCookies(self):
		"""
		返回浏览器里复制的整段 Cookie 字符串（分号分隔），便于整段替换、改 SSO 等。
		子类可重写；运行期也可直接改 self.cookie_string。
		"""
		return (
			'xxzlclientid=727a7e1d-08d2-47df-b164-1774582600521; '
			'xxzlxxid=pfmxbuGWwvZwEB4IxXDh0xpmec1F5H9At+0Hc6PEXO9nM75OzISzxl1oJp61soSV3S3g; '
			'xxzlbbid=pfmbRLroKy6LV9IcmyzqUYHsCt6N1loWpFzHkBUeu8+CL0nxGIKVGgkuerpN2dCgNmSZJ7uU/Auv6ncLzJ1B7/ZEgLCJtCvK5P9G6zT9I9cmShSuVj015MVUJ0RcswLKRDbm55FV/yIxNzc0NTgyNjAyOTMxODkz_1; '
			'dunCookie=69db01dcd6a19623cf0df9dedb0203ebc98397e67365e29bcffb6f75b2301d5b4e81cc4624e41c73; '
			'SSO_SESSION_ID=ST-1617369-qa9bpl17nas2ocWIuZdb-passport-58corp-com'
		)

	def login_get(self):
		cookies_names = ['xxzlclientid','xxzlxxid','wmda_uuid','wmda_new_uuid','wmda_visited_projects','ishare_sso_username','xxzlbbid','dunCookie']
		cookies = self.getCookiesStr(cookies_names)
		headers ={'Cookie':cookies,
				   'Host':'passport.58corp.com',
				  'User-Agent':self.user_agent
				}
		
		response = requests.request(method='get',url=self.login_url,headers=headers)
		if response.history:
			# 打印最后一个响应对象，即最初请求的响应对象
			print("重定向链接:", response.url)
			print("重定向来源:", response.history[0].url)
		else:
			print("没有重定向发生。")
		print("-----",response.headers)

	def get_pano_result_url(self,id):
		"""
		获取全景图结果 URL（兼容旧接口，返回日志 URL）
		
		:param id: 全景图 ID
		:return: (状态码, 日志URL)
		"""
		cookies_names = ['xxzlclientid','xxzlxxid','xxzlbbid','dunCookie','SSO_SESSION_ID']
		cookies = self.getCookiesStr(cookies_names)
		headers ={'Cookie':cookies,
				  'User-Agent':self.user_agent
				}
		
		data={'panoid':id,
			'source':'',
			'sdkVersion':'',
			'sdkType':'',
			'bids':'',
			'startDate':'',
			'endDate':'',
			'submit':'查询任务'}
		json_str = self.request_url(self.aihouse_url,headers=headers,data=data)
		res,log_url = self.parseJson(json_str)
		return res,log_url

	def parse_json_intermediate(self, js):
		"""
		解析接口返回的 JSON，取 CPU 任务（taskId=2）的中间结果包下载链接（temp 字段）。
		与 vrie.parseJson 一致，供下载 zip 使用。
		:return: (状态码, temp URL) 0 成功，1 无结果，2 解析失败
		"""
		try:
			data = json.loads(js)
			for task in data:
				if task.get('taskId') != 2:
					continue
				temp_url = task.get('temp')
				if not temp_url and task.get('infos'):
					try:
						infos = json.loads(task['infos'])
						temp_url = infos.get('temp')
					except (json.JSONDecodeError, TypeError):
						pass
				if temp_url:
					print(f"找到中间结果包链接: {temp_url}")
					return 0, temp_url
			print("未找到 taskId=2 的中间结果 temp 链接")
			return 1, None
		except json.JSONDecodeError as e:
			print(f"JSON 解析失败: {e}")
			return 2, None
		except Exception as e:
			print(f"解析中间结果 JSON 时发生错误: {e}")
			return 2, None

	def get_pano_intermediate_url(self, pano_id):
		"""
		查询全景任务并返回中间结果包（zip）下载 URL。
		"""
		cookies_names = ['xxzlclientid', 'xxzlxxid', 'xxzlbbid', 'dunCookie', 'SSO_SESSION_ID']
		cookies = self.getCookiesStr(cookies_names)
		headers = {'Cookie': cookies, 'User-Agent': self.user_agent}
		data = {
			'panoid': str(pano_id).strip(),
			'source': '',
			'sdkVersion': '',
			'sdkType': '',
			'bids': '',
			'startDate': '',
			'endDate': '',
			'submit': '查询任务',
		}
		json_str = self.request_url(self.aihouse_url, headers=headers, data=data)
		return self.parse_json_intermediate(json_str)

	def download_intermediate(self, url, save_path):
		"""
		流式下载中间结果包到本地（一般为 zip）。
		:return: 0 成功，1 失败
		"""
		try:
			headers = {'User-Agent': self.user_agent}
			os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
			with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
				resp.raise_for_status()
				with open(save_path, 'wb') as f:
					for chunk in resp.iter_content(chunk_size=65536):
						if chunk:
							f.write(chunk)
			print(f"下载完成: {save_path}")
			return 0
		except requests.exceptions.RequestException as e:
			print(f"下载失败: {e}")
			return 1
		except OSError as e:
			print(f"写入文件失败: {e}")
			return 1

	@staticmethod
	def unzip_and_remove_zip(zip_path):
		"""
		将 zip 解压到与压缩包同名的目录（去掉扩展名），成功后删除 zip。
		例如 out/138912665.zip -> 解压到 out/138912665/
		:return: 0 成功，1 失败
		"""
		zip_path = os.path.abspath(zip_path)
		if not os.path.isfile(zip_path):
			print(f'文件不存在: {zip_path}')
			return 1
		if not zipfile.is_zipfile(zip_path):
			print(f'不是有效的 zip，跳过: {zip_path}')
			return 1
		dest_dir = os.path.splitext(zip_path)[0]
		os.makedirs(dest_dir, exist_ok=True)
		try:
			with zipfile.ZipFile(zip_path, 'r') as zf:
				zf.extractall(dest_dir)
			os.remove(zip_path)
			print(f'已解压到: {dest_dir}，已删除: {zip_path}')
			return 0
		except zipfile.BadZipFile as e:
			print(f'解压失败（损坏的 zip）: {zip_path} {e}')
			return 1
		except OSError as e:
			print(f'解压或删除失败: {zip_path} {e}')
			return 1

	@staticmethod
	def read_pano_ids_from_excel(excel_path, sheet_name=0, column='pano_id'):
		"""
		从 Excel 读取 pano_id 列，去空、去重，返回字符串 ID 列表。
		兼容单元格为文本或数值（含 Excel 浮点 .0）。
		"""
		if pd is None:
			raise RuntimeError('需要安装 pandas 与 openpyxl: pip install pandas openpyxl')
		df = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=object)
		if column not in df.columns:
			alt = [c for c in df.columns if str(c).strip().lower() == column.lower()]
			if alt:
				column = alt[0]
			else:
				raise ValueError(f'Excel 中未找到列 {column!r}，实际列: {list(df.columns)}')
		ids = []
		for val in df[column]:
			if val is None or pd.isna(val):
				continue
			s = str(val).strip()
			if not s or s.lower() == 'nan':
				continue
			if re.match(r'^\d+\.0$', s):
				s = s[:-2]
			elif isinstance(val, float) and val == int(val):
				s = str(int(val))
			ids.append(s)
		# 去重且保持顺序
		seen = set()
		out = []
		for i in ids:
			if i not in seen:
				seen.add(i)
				out.append(i)
		return out

	@staticmethod
	def _sort_pano_ids_for_state(ids_set):
		def sort_key(x):
			try:
				return (0, int(x))
			except ValueError:
				return (1, x)
		return sorted(ids_set, key=sort_key)

	@staticmethod
	def load_downloaded_ids(state_path):
		"""从状态文件读取已完成的 pano_id 集合；文件不存在或损坏则返回空集。"""
		if not state_path or not os.path.isfile(state_path):
			return set()
		try:
			with open(state_path, 'r', encoding='utf-8') as f:
				data = json.load(f)
			raw = data.get('downloaded', [])
			return {str(x).strip() for x in raw if str(x).strip()}
		except (json.JSONDecodeError, OSError, TypeError):
			return set()

	@staticmethod
	def save_downloaded_ids(state_path, ids_set):
		"""原子写入已下载 pano_id 列表（不依赖本地是否仍有 zip/目录）。"""
		abs_path = os.path.abspath(state_path)
		dname = os.path.dirname(abs_path)
		if dname:
			os.makedirs(dname, exist_ok=True)
		payload = {'downloaded': Url._sort_pano_ids_for_state(ids_set)}
		tmp = abs_path + '.tmp'
		with open(tmp, 'w', encoding='utf-8') as f:
			json.dump(payload, f, ensure_ascii=False, indent=2)
		os.replace(tmp, abs_path)

	def batch_download_from_excel(
		self,
		excel_path,
		out_dir,
		sheet_name=0,
		unzip_after=False,
		state_file=None,
		force_redownload=False,
	):
		"""
		解析 Excel 的 pano_id 列，逐个拉取中间结果链接并下载到 out_dir。
		文件名: {pano_id}.zip（若 URL 带扩展名则尽量沿用）。
		unzip_after 为 True 时，下载成功后解压到同名子目录并删除 zip。
		已记录在 state_file 中的 pano_id 会跳过下载（文件移走也仍视为已下载）；
		force_redownload 为 True 时忽略记录仍重新下载，成功后仍会写回记录。
		"""
		if state_file is None:
			state_file = os.path.join(out_dir, 'pano_download_state.json')
		downloaded = self.load_downloaded_ids(state_file)
		pano_ids = self.read_pano_ids_from_excel(excel_path, sheet_name=sheet_name)
		os.makedirs(out_dir, exist_ok=True)
		print(f'共 {len(pano_ids)} 个 pano_id，输出目录: {out_dir}')
		print(f'下载记录文件: {os.path.abspath(state_file)}（已记录 {len(downloaded)} 个）')
		for pid in pano_ids:
			print(f'--- pano_id={pid} ---')
			if not force_redownload and pid in downloaded:
				print(f'已在下载记录中，跳过: {pid}')
				continue
			code, pack_url = self.get_pano_intermediate_url(pid)
			if code != 0 or not pack_url:
				print(f'跳过（无链接或失败）: {pid}, code={code}')
				continue
			ext = os.path.splitext(pack_url.split('?')[0])[1] or '.zip'
			save_path = os.path.join(out_dir, f'{pid}{ext}')
			if self.download_intermediate(pack_url, save_path) != 0:
				continue
			if unzip_after:
				if self.unzip_and_remove_zip(save_path) != 0:
					continue
			downloaded.add(pid)
			self.save_downloaded_ids(state_file, downloaded)

	def batch_unzip_in_dir(self, out_dir):
		"""
		将目录下所有 .zip 批量解压到同名子目录，解压成功后删除对应 zip。
		"""
		if not os.path.isdir(out_dir):
			print(f'目录不存在: {out_dir}')
			return
		zips = sorted(
			f for f in os.listdir(out_dir)
			if f.lower().endswith('.zip') and os.path.isfile(os.path.join(out_dir, f))
		)
		if not zips:
			print(f'未找到 zip: {out_dir}')
			return
		print(f'共 {len(zips)} 个 zip，目录: {out_dir}')
		for name in zips:
			print(f'--- {name} ---')
			self.unzip_and_remove_zip(os.path.join(out_dir, name))
	
	def get_pano_log_info(self, id):
		"""
		获取全景图的日志信息（完整流程：获取日志链接 -> 下载日志 -> 解析日志）
		提取 JSON 数据（位于 "dGltZQ== pj_total" 行的下一行）
		
		:param id: 全景图 ID
		:return: (状态码, 日志信息字典) 状态码: 0-成功, 1-日志不存在, 2-获取失败, 3-解析失败
		返回的日志信息字典包含：
		- log_url: 日志链接
		- json_found: 是否找到 JSON
		- json_content: JSON 原始字符串
		- json_data: 解析后的 JSON 对象（字典）
		- json_line_number: JSON 所在行号
		"""
		# 第一步：获取日志链接
		res, log_url = self.get_pano_result_url(id)
		if res != 0 or not log_url:
			return res, None
		
		# 第二步：获取日志内容
		res, log_content = self.get_log_content(log_url)
		if res != 0 or not log_content:
			return 2, None
		
		# 第三步：解析日志信息，提取 JSON
		try:
			log_info = self.parse_log_info(log_content)
			log_info['log_url'] = log_url
			
			# 打印提取结果
			if log_info.get('json_found'):
				print(f"成功提取 JSON 数据")
			else:
				error_msg = log_info.get('json_error', '未知错误')
				print(f"警告: 未找到 JSON 数据 - {error_msg}")
			
			return 0, log_info
		except Exception as e:
			print(f"解析日志信息时发生错误: {e}")
			return 3, None

	def parseJson(self,js):
		"""
		解析 JSON 结果，提取 CPU 任务（taskId=2）的日志链接
		
		:param js: JSON 字符串
		:return: (状态码, 日志URL) 状态码: 0-成功, 1-结果不存在, 2-解析失败
		"""
		try:
			data = json.loads(js)
			# 查找 taskId=2 的 CPU 任务
			for task in data:
				if task.get('taskId') == 2:
					# 优先使用 log 字段，如果没有则从 infos 中解析
					log_url = task.get('log')
					if not log_url and task.get('infos'):
						try:
							infos = json.loads(task['infos'])
							log_url = infos.get('log') or infos.get('log_url')
						except:
							pass
					
					if log_url:
						print(f"找到 CPU 任务日志链接: {log_url}")
						return 0, log_url
			
			print("未找到 taskId=2 的 CPU 任务")
			return 1, None
		except json.JSONDecodeError as e:
			print(f"JSON 解析失败: {e}")
			return 2, None
		except Exception as e:
			print(f"解析 JSON 时发生错误: {e}")
			return 2, None

	def get(self):
		pass
	def post(self):
		pass
		
	def request_url(self,url,headers=None,data=None):
		context = requests.request(method='post',url=url,headers=headers,data=data).text
		test = context.find("<textarea>")
		if test>0:
			context = context[test+ 13:]

		test = context.find("</textarea>")
		if test>0:
			context = context[0:test]

		if len(context) > 0:
			print("get url success")
		else:
			print("get url fail")	

		return context
	
	def get_log_content(self, log_url):
		"""
		通过日志链接获取日志内容
		
		:param log_url: 日志文件的 URL
		:return: (状态码, 日志内容) 状态码: 0-成功, 1-失败
		"""
		try:
			headers = {
				'User-Agent': self.user_agent
			}
			response = requests.get(log_url, headers=headers, timeout=30)
			response.raise_for_status()
			
			# 尝试使用 UTF-8 解码，如果失败则使用其他编码
			try:
				log_content = response.text
			except:
				log_content = response.content.decode('utf-8', errors='ignore')
			
			print(f"成功获取日志内容，长度: {len(log_content)} 字符")
			return 0, log_content
		except requests.exceptions.RequestException as e:
			print(f"获取日志失败: {e}")
			return 1, None
		except Exception as e:
			print(f"获取日志时发生错误: {e}")
			return 1, None
	
	def parse_log_info(self, log_content):
		"""
		从日志内容中提取 JSON 信息
		JSON 位于包含 "dGltZQ== pj_total" 的行的下一行
		
		:param log_content: 日志内容
		:return: 提取的信息字典，包含 JSON 数据
		"""
		info = {
			'full_content': log_content
		}
		
		lines = log_content.split('\n')
		
		# 提取 JSON 内容（在 "dGltZQ== pj_total" 行的下一行）
		json_content = None
		json_data = None
		json_line_number = None
		
		for i, line in enumerate(lines):
			line_stripped = line.strip()
			# 查找包含 "dGltZQ== pj_total" 的行
			if 'dGltZQ==' in line_stripped and 'pj_total' in line_stripped:
				# 取下一行作为 JSON
				if i + 1 < len(lines):
					json_line = lines[i + 1].strip()
					# 尝试解析为 JSON
					if json_line.startswith('{') and json_line.endswith('}'):
						try:
							json_data = json.loads(json_line)
							json_content = json_line
							json_line_number = i + 2  # 行号从1开始
							info['json_found'] = True
							print(f"成功提取 JSON 内容（第 {json_line_number} 行）")
							break
						except json.JSONDecodeError as e:
							print(f"JSON 解析失败: {e}")
							info['json_found'] = False
							info['json_error'] = str(e)
							json_line_number = i + 2
					else:
						info['json_found'] = False
						info['json_error'] = "下一行不是有效的 JSON 格式"
						json_line_number = i + 2
		
		if json_content:
			info['json_content'] = json_content
			info['json_data'] = json_data
			info['json_line_number'] = json_line_number
		else:
			info['json_content'] = ''
			info['json_data'] = None
			info['json_line_number'] = None
			if 'json_found' not in info:
				info['json_found'] = False
				info['json_error'] = "未找到包含 'dGltZQ== pj_total' 的行"
		
		return info

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='从 Excel 的 pano_id 列批量下载中间结果包，或调试单条日志 JSON')
	parser.add_argument('--excel', type=str, help='Excel 路径（含 pano_id 列）')
	parser.add_argument('--out-dir', type=str, default='middle_result_downloads', help='中间结果包保存目录')
	parser.add_argument('--sheet', default=0, help='工作表名或索引，默认 0')
	parser.add_argument('--log-demo', type=str, metavar='PANO_ID', help='仅调试：拉取该 pano 的日志并解析 JSON，不下载包')
	parser.add_argument('--unzip', action='store_true', help='下载成功后批量解压并删除 zip（仅与 --excel 一起使用）')
	parser.add_argument('--unzip-dir', type=str, metavar='DIR', help='仅解压：对该目录下所有 .zip 解压到同名文件夹并删除 zip（不下载）')
	parser.add_argument('--state-file', type=str, metavar='PATH', help='已下载 pano_id 记录 JSON，默认: <out-dir>/pano_download_state.json')
	parser.add_argument('--force', action='store_true', help='忽略下载记录，仍重新下载并更新记录')
	args = parser.parse_args()
	test = Url()

	if args.unzip_dir:
		test.batch_unzip_in_dir(args.unzip_dir)
	elif args.excel:
		sheet = args.sheet
		if isinstance(sheet, str) and sheet.isdigit():
			sheet = int(sheet)
		test.batch_download_from_excel(
			args.excel,
			args.out_dir,
			sheet_name=sheet,
			unzip_after=args.unzip,
			state_file=args.state_file,
			force_redownload=args.force,
		)
	elif args.log_demo:
		res, log_info = test.get_pano_log_info(args.log_demo)
		if res == 0:
			print(f"\n日志信息:")
			print(f"日志URL: {log_info.get('log_url')}")
			if log_info.get('json_found'):
				json_line_number = log_info.get('json_line_number', 'N/A')
				print(f"\nJSON 数据（第 {json_line_number} 行）:")
				print("=" * 60)
				json_data = log_info.get('json_data', {})
				if json_data:
					print(json.dumps(json_data, ensure_ascii=False, indent=2))
				else:
					print(log_info.get('json_content', ''))
				print("=" * 60)
			else:
				print(f"\n未找到 JSON 数据")
				if log_info.get('json_error'):
					print(f"错误: {log_info.get('json_error')}")
		else:
			print(f"获取日志信息失败，状态码: {res}")
	else:
		parser.print_help()
		print('\n示例: python url.py --excel ./data.xlsx --out-dir ./downloads')
		print('      python url.py --excel ./data.xlsx --out-dir ./downloads --unzip')
		print('      python url.py --unzip-dir ./downloads')
		print('      python url.py --excel ./data.xlsx --state-file ~/vr_pano_state.json')
		print('      python url.py --log-demo 136415252')