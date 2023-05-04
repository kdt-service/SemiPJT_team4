import scrapy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from SearchScraper.items import SearchscraperItem
from SearchScraper.cleanser import cleansing
import re

class SearchSpiderSpider(scrapy.Spider):
    name = "search_spider"
    allowed_domains = ["search.naver.com", "news.naver.com"]
    start_urls = ["http://search.naver.com/"]
    # 자꾸 m.serch.naver.com으로 redirection 되길래 추가
    handle_httpstatus_list = [302, 403]

    keyword = '하이닉스'
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    today, yesterday
    # start_date = (2023, 4, 25)
    # end_date = (2023, 5, 3)

    URL_FORMAT = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query={}&sort=0&photo=0&field=0&pd=3&ds={}&de={}&start={}'

    def start_requests(self):
        dates = pd.date_range(end=self.today, start=self.yesterday).strftime('%Y.%m.%d').tolist()
        for date in dates:
            target_url = self.URL_FORMAT.format(
                            self.keyword, 
                            date, date,
                            1
                            )
            yield scrapy.Request(url=target_url, callback=self.parse_url, meta={'page':1, 'keyword':self.keyword})

    def parse_url(self, response):
        # 검색결과 없다는 칸이 있으면 중지
        if response.xpath('//*[@class="api_noresult_wrap"]').extract() != []:
            return
        
        urls = response.xpath('//*[@class="bx"]/div/div/div[1]/div[2]/a[2]/@href').extract()

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse, meta={**response.meta})

        with open('./test_url_list.txt', 'a') as f:
            for url in urls:
                f.write(url+'\n')

        page = response.meta.pop('page') + 1
        # 400페이지 넘긴다면 중지
        if page > 400:
            return
        
        target_url = re.sub('start\=\d+', f'start={10*(page-1)+1}', response.url)
        yield scrapy.Request(url=target_url, callback=self.parse_url, meta={**response.meta, 'page':page})

    def parse(self, response):
        item = SearchscraperItem()
        item['keyword'] = response.meta['keyword']
        item['title'] = response.xpath('//*[@id="title_area"]/span/text()').extract()
        item['content'] = cleansing(' '.join(response.xpath('//*[@id="dic_area"]//text()').extract()))
        try:
            item['writer'] = response.css('.byline_s::text').get().strip().split(' ')[0].split('(')[0]
        except:
            item['writer'] = None
        item['writed_at'] = response.css('.media_end_head_info_datestamp_time::attr(data-date-time)').get()
        item['url'] = response.url
        yield item
