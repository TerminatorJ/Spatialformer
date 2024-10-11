from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time

# Path to your chrome driver
# CHROME_DRIVER_PATH = '/path/to/chromedriver'


# service = Service(CHROME_DRIVER_PATH)
# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Useful for scripts running in environments without GUI
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Use a Service object created for Chrome with WebDriverManager
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)
import pdb; pdb.set_trace()

# driver = webdriver.Chrome(service=service)

try:
    # Open the website
    url = "https://www.10xgenomics.com/datasets?query=Xenium&page=1&configure%5BhitsPerPage%5D=50&configure%5BmaxValuesPerFacet%5D=1000&refinementList%5Bplatform%5D%5B0%5D=Xenium%20In%20Situ"
    driver.get(url)
    import pdb; pdb.set_trace()
    # Wait until the datasets are loaded
    wait = WebDriverWait(driver, 10)
    datasets = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div[data-test-id="dataset-card"]')))
    import pdb; pdb.set_trace()
    # Prepare lists to store information
    dataset_info = []
    download_links = []
    import pdb; pdb.set_trace()
    for dataset in datasets:
        # Open dataset details
        import pdb; pdb.set_trace()
        dataset_url = dataset.find_element(By.CSS_SELECTOR, 'a').get_attribute('href')
        driver.get(dataset_url)
        time.sleep(2)  # wait for the page to load completely

        # Gather overview data
        overview_data = driver.find_element(By.CSS_SELECTOR, 'div[data-test-id="dataset-overview"]').text

        # Extract the number of cells/transcripts
        cells_or_transcripts = None
        # You might need to parse overview_data to extract this info
        # Example: cells_or_transcripts = parse_overview_info(overview_data)

        # Find first download link
        batch_download_button = driver.find_element(By.XPATH, '//button[text()="Batch download"]')
        batch_download_button.click()
        time.sleep(1)  # allow loading
        import pdb; pdb.set_trace()
        download_code = driver.find_element(By.XPATH, '(//code[contains(text(), "curl -O")])[1]').text
        # Split to get the URL
        zip_file_url = download_code.split(' ')[2]

        # Store the results
        dataset_info.append({'Dataset Name': dataset_url.split("/")[-1], 'Cells/Transcripts': cells_or_transcripts})
        download_links.append(zip_file_url)
        import pdb; pdb.set_trace()
        # Navigate back to the main page
        driver.back()
        time.sleep(2)  # Delay to ensure page loaded

    # Convert to DataFrame and save
    import pdb; pdb.set_trace()
    df = pd.DataFrame(dataset_info)
    df.to_csv('datasets_info.csv', index=False)
    import pdb; pdb.set_trace()
    # Save URLs to a text file
    with open('download_links.txt', 'w') as file:
        for link in download_links:

            file.write(f"{link}\n")

finally:
    driver.quit()
