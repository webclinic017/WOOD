from lxml import html, etree
import requests
from selenium import webdriver
import time
import pandas as pd
import pickle
import os

class PagesJaunesScrapper():
    """
        firms is a dict :
            key : raison-sociale (en minuscule)
            value : ville-codepostal (en minuscule)
    """
    def __init__(self, driver):
        self.driver = driver
        self.driver.get(
            'https://www.pagespro.com/recherche/Paris/actalians'
        )

    def get_result_tree_from_firm_name(self, firm_siret, firm_name, town):
        """ return siret, raison sociale, phone
        """
        # time.sleep(2)
        time.sleep(0.2)
        queryWhat = self.driver.find_element_by_xpath("//input[@name='queryWhat']")
        queryWhat.clear()
        queryWhat.send_keys(firm_name)

        queryWhere = self.driver.find_element_by_xpath("//input[@name='queryWhere']")
        queryWhere.clear()
        queryWhere.send_keys(town)

        search_button = self.driver.find_element_by_xpath("//button[@class='adp-searchForm__submit']")
        search_button.click()
        time.sleep(0.1)
        try:
            phone_button = self.driver.find_element_by_id("tel_1")
            phone_button.click()
            time.sleep(0.2)
            source_page = self.driver.page_source
            html_source_page = html.fromstring(source_page)
            phone = html_source_page.xpath("//span[@id='tel_1']//a/text()")[0]

            clicking_title = self.driver.find_element_by_xpath("//span[contains(@class, 'adp-listingResultHeader__name')]")
            clicking_title.click()
            time.sleep(0.2)
        except:
            return "NC", "NC", "NC"
        try:
            source_page = self.driver.page_source
            information_html_source_page = html.fromstring(source_page)
            siret = information_html_source_page.xpath("//div[@class='adp-detailsSection']//div[text()='SIRET']/following-sibling::div[1]/text()")[0]
            raison_sociale = information_html_source_page.xpath("//h1[@class='adp-h2  adp-detailsHeader__title']/text()")[0]
        except:
            return 'NC', 'NC', phone
        return siret, raison_sociale, phone


if __name__ == "__main__":
    try:
        with open ('set_siret.pkl', 'rb') as fp:
            set_siret = pickle.load(fp)
    except:
        set_siret = set()

    df = pd.read_csv('liste_entreprises_pour_scrapping.csv', sep=';')
    final_df = pd.DataFrame(columns=['SIRET', 'RAISON SOCIALE', 'VILLE', 'PHONE NUMBER FOUND', 'SIRET FOUND', 'RAISON SOCIALE FOUND'])
    print(df.head())

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")

    driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver', chrome_options=chrome_options)
    driver.set_window_size(1120, 550)
    time.sleep(0.2)
    pages_jaunes_scrapper = PagesJaunesScrapper(driver=driver)

    nb_iterations = 0
    nb_file = 0
    for filename in os.listdir('FINAL_FILES/'):
        nb_file += 1
    nb_file -= 1
    print(nb_file)

    # iterate over rows with iterrows()
    for index, row in df.iterrows():
        print(row['SIRET (établissement siège)'])
        if row['SIRET (établissement siège)'] in set_siret:
            print("pass")
            continue
        siret_found, raison_sociale_found, phone_found = pages_jaunes_scrapper.get_result_tree_from_firm_name(row['SIRET (établissement siège)'], row['Raison sociale'], row['Ville'])
        final_df = final_df.append({
            'SIRET': row['SIRET (établissement siège)'],
            'RAISON SOCIALE': row ['Raison sociale'],
            'VILLE': row['Ville'],
            'PHONE NUMBER FOUND': phone_found,
            'SIRET FOUND': siret_found,
            'RAISON SOCIALE FOUND': raison_sociale_found
            }, ignore_index=True
        )
        nb_iterations += 1
        set_siret.add(row['SIRET (établissement siège)'])

        if nb_iterations % 100 == 0:
            final_df.to_csv('FINAL_FILES/final_df' + str(nb_file) + '.csv', index=False, sep=';')
            final_df = pd.DataFrame(columns=['SIRET', 'RAISON SOCIALE', 'VILLE', 'PHONE NUMBER FOUND', 'SIRET FOUND', 'RAISON SOCIALE FOUND'])
            nb_file += 1
            print(nb_iterations)
            with open('set_siret.pkl', 'wb') as fp:
                pickle.dump(set_siret, fp)

    final_df.to_csv('FINAL_FILES/final_df_final.csv', index=False, sep=';')
    '''firms = [
        ['82118689700020', 'mayer-prezioso', 'paris-75'],
    ]
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")

    driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver', chrome_options=chrome_options)
    driver.set_window_size(1120, 550)

    """# Email insert
    driver.get("https://accounts.google.com/signin/v2/identifier?service=mail&passive=true&rm=false&continue=https%3A%2F%2Fmail.google.com%2Fmail%2F&ss=1&scc=1&ltmpl=default&ltmplcache=2&emr=1&osid=1&flowName=GlifWebSignIn&flowEntry=ServiceLogin")  #URL of email page
    username = driver.find_element_by_id("identifierId")
    username.send_keys("EMAIL")
    driver.find_element_by_id("identifierNext").click()
    time.sleep(1)

    # Password Insert
    password = driver.find_element_by_name("password")
    password.send_keys("PASSWORD$")
    driver.find_element_by_id("passwordNext").click()"""
    time.sleep(0.2)
    pages_jaunes_scrapper = PagesJaunesScrapper(driver=driver)

    for firm in firms:
        pages_jaunes_scrapper.get_result_tree_from_firm_name(firm[0], firm[1], firm[2])'''
