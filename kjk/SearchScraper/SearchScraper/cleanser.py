import re

# 통합 클렌징 코드
def cleansing(text:str) -> str:

    # 특수기호 제거
    # text = re.sub('[▶△▶️◀️▷■◆●]', '', text)
    # ·ㆍ■◆△▷▶▼�"'…※↑↓▲☞ⓒ⅔
    
    text = text.replace('“','"').replace('”','"')
    text = text.replace("‘","'").replace("’","'")
    text = text.replace('·',', ').replace('ㆍ',', ').replace('…','...')

    # 인코딩오류 해결 (공백으로 치환)
    text = re.sub('[\xa0\u2008\u2190]', ' ', text)

    # URL제거를 위해 필요없는 문구 처리
    text = text.replace('https?\:\/*', '')
    # 이메일 처리, URL 제거
    # '[\w\.-]+(\@|\.)[\w\.-]+\.[\w\.]+'
    text = re.sub('[\w\-]+(\@|\.)[\w\-\.]+[^\s]*', '', text)

    # 카피라이트, 트위터 아이디 제거
    text = re.sub('[^ ]*[ⓒ@][^ ]*', '', text)

    # ., 공백, 줄바꿈 여러개 제거 
    # \s -> 공백( ), 탭(\t), 줄바꿈(\n)
    text = re.sub('[\.]{2,}', '.', text)
    text = re.sub('[\t]+', ' ', text)
    text = re.sub('[ ]{2,}', ' ', text)
    text = re.sub('(\\n)+[(\\n) ]+', '\n', text)
    text = text.strip()

    return text