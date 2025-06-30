import requests
from bs4 import BeautifulSoup

# 目標網址（可替換成你想爬的網站）
url = 'https://hgmoldmaker.weebly.com/'

# 發送 HTTP GET 請求
response = requests.get(url)

# 確認請求成功
if response.status_code == 200:
    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 以 CSS 選擇器找出想要的元素，例如所有的 <a> 標籤
    links = soup.select('a')
    
    # 印出每個連結的文字和網址
    for link in links:
        text = link.get_text(strip=True)
        href = link.get('href')
        print(f'Text: {text}, Link: {href}')
else:
    print(f'無法取得網頁，狀態碼：{response.status_code}')
