import numpy
from lxml import html
import requests
import re
import urllib
import os
import sys
import requests
import json
from requests.exceptions import ConnectionError
import time



def download_images(show_observations_list,wanted_year):
	
	gbif_information_d = {}


	dictionary_file = open('observations_'+str(wanted_year)+'.json','a')
	dictionary_file.write('[')
	
	
	try:
		for ob in show_observations_list:
			
			print 'observation_number: '+str(ob)
			show_observations_url = 'http://mushroomobserver.org'+ob
			page_observation=None
			
			try:
			    page_observation = requests.get(show_observations_url)
			except ConnectionError:
			    time.sleep(10)
			    page_observation = requests.get(show_observations_url)
			tree_observation = html.fromstring(page_observation.content)
			name = tree_observation.xpath('//span[@id="title-caption"]//i/text()')


			if len(name)>0:
			    date = tree_observation.xpath('//div[@id="observation_when"]/strong/text()')[0]
			    date_split = date.split('-')
			    year = date_split[0]
			    month = int(date_split[1])
			    day = int(date_split[2])
			    print 'date: '+str(date)


			    if int(year)<int(wanted_year):
					pass


			    elif int(year)>int(wanted_year):
			    	pass

			    else:
			    	name = name[0]
			        name_id = tree_observation.xpath('//div[@class="list-group"]//div[@class="small"]/a/@href')[0]
			        name_page=""
			        try:
			            name_page = requests.get('http://mushroomobserver.org'+name_id).content
			        except ConnectionError:
			            time.sleep(10)
			            name_page = requests.get('http://mushroomobserver.org'+name_id).content

			        if 'Preferred Synonym(s)' in name_page:
			            tree_names = html.fromstring(name_page.split('Preferred Synonym(s):')[1].split('</a>')[0])
			            preffered_name = tree_names.xpath('//a/b/i/text()')[0]
			            name = preffered_name

			        name = name.replace('"','')

			        images = tree_observation.xpath('//div[@class="show_images list-group"]//img/@alt')
			        where = tree_observation.xpath('//div[@id="observation_where"]//a/@href')[0]
			        who = tree_observation.xpath('//div[@id="observation_who"]//a/@href')[0]

			        gbif_response = None
			        if name in gbif_information_d:
			        	gbif_response = gbif_information_d[name]
			        else:
			        	url = 'http://api.gbif.org/v1/species/match?name='+name
			        	try:
			        		gbif_response = requests.get(url).json()
			        		gbif_information_d[name] = gbif_response
			        	except:
			        		print 'not able to get gbif information '+name

			        if len(images)>0:   
			            for i in range(len(images)):
			                d = {}
			                image = images[i]
			                d['label'] = name
			                d['date'] = date
			                d['image_id'] = image
			                d['image_url'] = 'http://mushroomobserver.org/images/320/' + image
			                d['location'] = where
			                d['user'] = who
			                d['gbif_info'] = gbif_response
			                d['observation'] = ob
				                
			                if i==0:
			                    d['thumbnail']=1
			                else:
			                    d['thumbnail']=0

			            	json.dump(d,dictionary_file)
			                dictionary_file.write(',')

	except KeyboardInterrupt:
		print 'closing json file'
		dictionary_file.seek(-1, os.SEEK_END)
		dictionary_file.truncate()
		dictionary_file.write(']')
		dictionary_file.close()	
		return False
    
    
	print 'closing json file'
	dictionary_file.seek(-1, os.SEEK_END)
	dictionary_file.truncate()
	dictionary_file.write(']')
	dictionary_file.close()
	return True


def scrape_observations(year):
	#get all the nodes that point to observations

	page = requests.get('http://mushroomobserver.org/sitemap/index.html')
	tree = html.fromstring(page.content)
	observations_list = tree.xpath('//a[contains(@href, "observations")]/@href')
	observations_list = observations_list[::-1]
	

	for index in range(len(observations_list)):
		observations = observations_list[index]
		page = requests.get('http://mushroomobserver.org/sitemap/'+observations)
		tree = html.fromstring(page.content)
		obs =  tree.xpath('//a[contains(@href, "show_observation")]/@href')
		return_value = download_images(obs,year)
		if return_value==False:
			break





year = int(sys.argv[1])
scrape_observations(year)
