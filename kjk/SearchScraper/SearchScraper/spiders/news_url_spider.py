import scrapy
from SearchScraper.items import SearchscraperItem
from SearchScraper.cleanser import cleansing

class NewsUrlSpiderSpider(scrapy.Spider):
    name = "news_url_spider"
    allowed_domains = ["news.naver.com"]
    start_urls = ["http://news.naver.com/"]

    keyword = '하이닉스'

    def start_requests(self):
        with open('./url_list.txt', 'r') as f:
            urls = [url.strip() for url in f.readlines()]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        item = SearchscraperItem()
        item['keyword'] = self.keyword
        item['title'] = response.xpath('//*[@id="title_area"]/span/text()').extract()
        item['content'] = cleansing(' '.join(response.xpath('//*[@id="dic_area"]//text()').extract()))
        try:
            item['writer'] = response.css('.byline_s::text').get().strip().split(' ')[0].split('(')[0].replace('기자', '')
        except:
            item['writer'] = None
        item['writed_at'] = response.css('.media_end_head_info_datestamp_time::attr(data-date-time)').get()
        item['url'] = response.url

        if item['title'] == []:
            with open('./error_urls', 'a') as f:
                f.write(response.url + '\n')
            return
        
        yield item