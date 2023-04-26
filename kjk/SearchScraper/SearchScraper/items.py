# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class SearchscraperItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    keyword = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()
    writer = scrapy.Field()
    writed_at = scrapy.Field()
    url = scrapy.Field()
    pass
