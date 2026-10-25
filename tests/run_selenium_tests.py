import time
import os
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

workspace_root = Path(r"C:\Users\ragha\OneDrive - vit.ac.in\Subject\Sem 5\Software Engineering\brain-tumor-diagnostic-workspace")
output_dir = workspace_root / "tests" / "selenium_results"
output_dir.mkdir(parents=True, exist_ok=True)

test_image_path = workspace_root / "data" / "processed" / "test" / "images" / "TCGA_HT_7608_19940304_TCGA_HT_7608_19940304_12.png"

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1920,1080")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--force-device-scale-factor=1.5")

print("===================================================================")
print(" STARTING SELENIUM AUTOMATED FUNCTIONAL TEST SUITE")
print(" System: 2D Brain Tumor Diagnostic Workspace")
print(" Student: Raghava G Vedantam (24BRS1250)")
print("===================================================================")

driver = webdriver.Chrome(options=opts)
driver.set_window_size(1920, 1080)
wait = WebDriverWait(driver, 20)

test_results = []

try:
    # ── TC-01: Page Initialization ──────────────────────────────────────────
    print("\n[RUNNING] TC-01: Web Application Initialization & DOM Rendering")
    driver.get("http://localhost:8501")
    time.sleep(4)
    
    title = driver.title
    print(f"  --> Page Title: {title}")
    assert "Brain Tumor" in title or "Detector" in title or "Streamlit" in title, "Page title mismatch"
    driver.save_screenshot(str(output_dir / "01_homepage_initialization.png"))
    test_results.append(("TC-01", "Web Application Initial Load", "HTTP 200 & DOM Render", "Title & UI rendered", "PASS"))
    print("  --> [PASS] TC-01 Passed successfully.")

    # ── TC-02: Admin Authentication Gate ────────────────────────────────────
    print("\n[RUNNING] TC-02: Role Selection & Administrator Password Gate")
    selectboxes = driver.find_elements(By.CSS_SELECTOR, "[data-testid='stSelectbox']")
    if selectboxes:
        selectboxes[0].click()
        time.sleep(1)
        admin_opt = driver.find_element(By.XPATH, "//*[text()='Administrator']")
        admin_opt.click()
        time.sleep(2)
        
        # Test Invalid Password
        pwd_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='password']")
        assert len(pwd_inputs) > 0, "Password input field not displayed"
        pwd_input = pwd_inputs[0]
        pwd_input.send_keys("wrong_pwd" + Keys.RETURN)
        time.sleep(2)
        driver.save_screenshot(str(output_dir / "02_admin_auth_invalid.png"))
        assert "Incorrect password" in driver.page_source, "Failed to display incorrect password message"
        print("  --> Sub-test: Invalid password correctly rejected with error banner.")

        # Test Valid Password with React clearing
        pwd_input = driver.find_elements(By.CSS_SELECTOR, "input[type='password']")[0]
        pwd_input.send_keys(Keys.CONTROL + "a")
        pwd_input.send_keys(Keys.BACKSPACE)
        pwd_input.send_keys("1234" + Keys.RETURN)
        time.sleep(3)
        driver.save_screenshot(str(output_dir / "03_admin_auth_success.png"))
        assert "Admin access granted" in driver.page_source, "Failed to grant admin access"
        print("  --> Sub-test: Valid password accepted. Admin privileges granted.")
        test_results.append(("TC-02", "Role & Authentication Gate", "Password Guard (Admin: 1234)", "Passcode verified; access granted", "PASS"))
        print("  --> [PASS] TC-02 Passed successfully.")

    # ── TC-03: Functional MRI Scan Upload ────────────────────────────────────
    print("\n[RUNNING] TC-03: Functional MRI Scan File Upload")
    file_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
    assert len(file_inputs) > 0, "File uploader input element not found"
    file_input = file_inputs[0]
    
    print(f"  --> Uploading scan: {test_image_path.name}")
    file_input.send_keys(str(test_image_path.resolve()))
    
    print("  --> Waiting for Deep Learning inference pipeline...")
    time.sleep(7)
    
    driver.save_screenshot(str(output_dir / "04_mri_upload_and_inference.png"))
    test_results.append(("TC-03", "MRI Scan Upload Handling", "File Uploader Interface", f"Uploaded {test_image_path.name}", "PASS"))
    print("  --> [PASS] TC-03 Passed successfully.")

    # ── TC-04: Deep Learning Segmentation Panels ─────────────────────────────
    print("\n[RUNNING] TC-04: Multi-View Segmentation Visualization Verification")
    page_content = driver.page_source
    assert "Original MRI" in page_content, "Original MRI caption missing"
    assert "AI Overlay" in page_content, "AI Overlay caption missing"
    assert "Segmentation Mask" in page_content, "Segmentation Mask caption missing"
    
    images = driver.find_elements(By.CSS_SELECTOR, "div[data-testid='stImage'] img")
    print(f"  --> Rendered visual panel images: {len(images)}")
    assert len(images) >= 3, "Expected 3 diagnostic visual panels"
    
    test_results.append(("TC-04", "Deep Learning Segmentation", "U-Net ResNet34 Multi-View", f"{len(images)} diagnostic views generated", "PASS"))
    print("  --> [PASS] TC-04 Passed successfully.")

    # ── TC-05: Quantitative Clinical Biomarkers ──────────────────────────────
    print("\n[RUNNING] TC-05: Automated Radiological Biomarker Extraction")
    driver.execute_script("window.scrollTo(0, 550);")
    time.sleep(1.5)
    driver.save_screenshot(str(output_dir / "05_clinical_biomarkers_scroll.png"))
    
    page_text = driver.page_source.lower()
    found_biomarkers = []
    for marker in ["lesion area", "lesion pixels", "foci detected", "diameter", "rano", "burden", "clinical impression"]:
        if marker in page_text:
            found_biomarkers.append(marker)
    
    print(f"  --> Extracted clinical biomarkers: {found_biomarkers}")
    assert len(found_biomarkers) >= 4, "Missing expected quantitative biomarkers"
    test_results.append(("TC-05", "Quantitative Biomarker Engine", "RANO & Morphology Analysis", f"Computed {len(found_biomarkers)} clinical markers", "PASS"))
    print("  --> [PASS] TC-05 Passed successfully.")

    # ── TC-06: Clinical Impression & PDF Report Generation ───────────────────
    print("\n[RUNNING] TC-06: Clinical Impression Synthesis & PDF Report Dispatcher")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.5)
    driver.save_screenshot(str(output_dir / "06_clinical_report_download_section.png"))
    
    download_buttons = driver.find_elements(By.CSS_SELECTOR, "[data-testid='stDownloadButton']")
    print(f"  --> Found PDF Download button: {len(download_buttons) > 0}")
    assert len(download_buttons) > 0, "PDF Download button not rendered"
    
    test_results.append(("TC-06", "Clinical Report Dispatcher", "ReportLab PDF Engine", "Report compiled & download active", "PASS"))
    print("  --> [PASS] TC-06 Passed successfully.")

except Exception as err:
    print(f"\n[FAIL] Test Execution Error Encountered: {err}")
    driver.save_screenshot(str(output_dir / "error_snapshot.png"))
    test_results.append(("TC-ERR", "Test Failure", "Exception caught", str(err)[:60], "FAIL"))
finally:
    driver.quit()

# Write formatted execution summary
summary_file = output_dir / "selenium_test_report.txt"
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("========================================================================================\n")
    f.write("      AUTOMATED FUNCTIONAL TEST EXECUTION REPORT (SELENIUM WEBDRIVER)                   \n")
    f.write("      PROJECT: 2D BRAIN TUMOR DIAGNOSTIC WORKSPACE                                      \n")
    f.write("      STUDENT: RAGHAVA G VEDANTAM (24BRS1250)                                           \n")
    f.write(f"      EXECUTION TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}                              \n")
    f.write("========================================================================================\n\n")
    f.write(f"{'Test ID':<10} | {'Test Scenario':<32} | {'Target Component':<26} | {'Status':<8}\n")
    f.write("-" * 88 + "\n")
    for r in test_results:
        f.write(f"{r[0]:<10} | {r[1]:<32} | {r[2]:<26} | {r[4]:<8}\n")
    f.write("-" * 88 + "\n")
    total = len(test_results)
    passed = sum(1 for r in test_results if r[4] == "PASS")
    f.write(f"\nTotal Test Cases: {total} | Passed: {passed} | Failed: {total - passed} | Success Rate: {(passed/total)*100:.1f}%\n")

print("\n===================================================================")
print(" SELENIUM TEST EXECUTION COMPLETED")
print(f" Results logged at: {summary_file}")
print("===================================================================")
